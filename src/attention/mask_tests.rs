//! Exclusion-plane tests: bottom-right causal oracle, unions, and padding planes.

use burn::{
    backend::Flex,
    tensor::{Bool, DType, Tensor},
};

use super::*;

/// Deterministic CPU backend for mask construction.
type TestBackend = Flex<f32, i32>;

/// Reads a Boolean tensor into host memory.
fn bools<const D: usize>(tensor: Tensor<TestBackend, D, Bool>) -> Vec<bool> {
    tensor.into_data().to_vec::<bool>().unwrap()
}

/// Every causal plane matches `key > query + K - Q`, including `Q > K`.
#[test]
fn causal_plane_matches_bottom_right_oracle() {
    let device = Default::default();
    for queries in 1..=5 {
        for keys in 1..=5 {
            let plane = bottom_right_causal_plane::<TestBackend>(queries, keys, &device).unwrap();
            assert_eq!(plane.dims(), [1, 1, queries, keys]);
            let actual = bools(plane);
            for query in 0..queries {
                for key in 0..keys {
                    let excluded = key as i64 > query as i64 + keys as i64 - queries as i64;
                    assert_eq!(actual[query * keys + key], excluded, "q={queries} k={keys}");
                }
            }
        }
    }
}

/// The expanded causal mask repeats one plane across batch and heads.
#[test]
fn causal_mask_expands_across_batch_and_heads() {
    let device = Default::default();
    let mask = bottom_right_causal_mask::<TestBackend>(2, 3, 2, 4, &device).unwrap();
    assert_eq!(mask.dims(), [2, 3, 2, 4]);
    assert_eq!(
        bools(mask),
        [false, false, false, true, false, false, false, false].repeat(6)
    );
}

/// Unions broadcast size-one axes and treat `None` as an empty exclusion.
#[test]
fn unions_broadcast_and_respect_missing_operands() {
    let device = Default::default();
    let rows = Tensor::<TestBackend, 4, Bool>::from_bool([[[[true], [false]]]].into(), &device);
    let columns =
        Tensor::<TestBackend, 4, Bool>::from_bool([[[[false, true, false]]]].into(), &device);
    let union = union_exclusions(Some(rows.clone()), Some(columns)).unwrap();
    assert_eq!(union.dims(), [1, 1, 2, 3]);
    assert_eq!(bools(union), [true, true, true, false, true, false]);
    assert_eq!(
        bools(union_exclusions(Some(rows.clone()), None).unwrap()),
        bools(rows.clone())
    );
    assert_eq!(
        bools(union_exclusions(None, Some(rows.clone())).unwrap()),
        bools(rows)
    );
    assert!(union_exclusions::<TestBackend>(None, None).is_none());
}

/// Padding planes exclude invalid query rows and key columns and validate extents.
#[test]
fn padding_plane_marks_invalid_tokens_and_rejects_wrong_extents() {
    let device = Default::default();
    let query = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false]].into(),
        &device,
    ));
    let key = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[false, true, true]].into(),
        &device,
    ));
    let plane = padding_plane(Some(&query), Some(&key), 1, 2, 3)
        .unwrap()
        .unwrap();
    assert_eq!(plane.dims(), [1, 1, 2, 3]);
    assert_eq!(bools(plane), [true, false, false, true, true, true]);
    assert!(
        padding_plane::<TestBackend>(None, None, 1, 2, 3)
            .unwrap()
            .is_none()
    );
    let error = padding_plane(Some(&key), None, 1, 2, 3).unwrap_err();
    assert!(error.to_string().contains("query_mask"));
    let error = padding_plane(None, Some(&query), 1, 2, 3).unwrap_err();
    assert!(error.to_string().contains("key_mask"));
}

/// Positions are bottom-right aligned and lengths beyond `i64` fail with context.
#[test]
fn positions_are_aligned_and_lengths_are_checked() {
    let device = Default::default();
    let positions = aligned_query_positions::<TestBackend>(2, 5, &device).unwrap();
    assert_eq!(positions.into_data().to_vec::<i32>().unwrap(), [3, 4]);
    let negative = aligned_query_positions::<TestBackend>(3, 1, &device).unwrap();
    assert_eq!(negative.into_data().to_vec::<i32>().unwrap(), [-2, -1, 0]);
    assert_eq!(
        key_positions::<TestBackend>(3, &device)
            .unwrap()
            .into_data()
            .to_vec::<i32>()
            .unwrap(),
        [0, 1, 2]
    );
    #[cfg(target_pointer_width = "64")]
    {
        let error = signed_length(i64::MAX as usize + 1, "too long").unwrap_err();
        assert!(error.to_string().contains("too long"));
    }
    let mask = Tensor::<TestBackend, 4, Bool>::zeros([1, 1, 1, 1], &device);
    assert_eq!(expand_to(mask.clone(), [1, 1, 1, 1]).dims(), [1, 1, 1, 1]);
    assert_eq!(expand_to(mask, [2, 1, 3, 1]).dims(), [2, 1, 3, 1]);
}

/// Integer limits follow each dtype, with a conservative `i32` limit for non-integers.
#[test]
fn position_limits_follow_the_integer_dtype() {
    assert_eq!(position_limit(DType::I64), i64::MAX);
    assert_eq!(position_limit(DType::U64), i64::MAX);
    assert_eq!(position_limit(DType::I32), i64::from(i32::MAX));
    assert_eq!(position_limit(DType::U32), i64::from(u32::MAX));
    assert_eq!(position_limit(DType::I16), i64::from(i16::MAX));
    assert_eq!(position_limit(DType::U16), i64::from(u16::MAX));
    assert_eq!(position_limit(DType::I8), i64::from(i8::MAX));
    assert_eq!(position_limit(DType::U8), i64::from(u8::MAX));
    assert_eq!(position_limit(DType::F32), i64::from(i32::MAX));
}

/// Position arithmetic is checked against the device's `i32` dtype before allocating.
#[test]
fn oversized_positions_fail_before_allocation() {
    let device = Default::default();
    ensure_positions_fit::<TestBackend>(&device, i64::from(i32::MAX)).unwrap();
    let error = ensure_positions_fit::<TestBackend>(&device, i64::from(i32::MAX) + 1).unwrap_err();
    assert!(error.to_string().contains("attention.positions"), "{error}");
    // `2^31 + 1` keys would need position `2^31`, which `i32` cannot hold.
    let keys = (1usize << 31) + 1;
    assert!(key_positions::<TestBackend>(keys, &device).is_err());
    assert!(aligned_query_positions::<TestBackend>(1, keys, &device).is_err());
    assert!(bottom_right_causal_plane::<TestBackend>(2, keys, &device).is_err());
}

/// Token-mask validation reads only metadata and names the offending mask.
#[test]
fn token_mask_checks_report_extents() {
    let device = Default::default();
    let mask = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, false, true]].into(),
        &device,
    ));
    check_token_mask(&mask, "key_mask", 1, 3).unwrap();
    let error = check_token_mask(&mask, "key_mask", 2, 3).unwrap_err();
    assert!(error.to_string().contains("key_mask"), "{error}");
    assert!(error.to_string().contains("[2, 3]"), "{error}");
}

/// The test-only audit keeps the largest recorded plane until reset.
#[test]
fn plane_audit_tracks_the_largest_plane() {
    plane_audit::reset();
    audit_plane([1, 2, 3, 4]);
    audit_plane([1, 1, 2, 2]);
    assert_eq!(plane_audit::largest(), 24);
    plane_audit::reset();
    assert_eq!(plane_audit::largest(), 0);
    union_exclusions::<TestBackend>(
        Some(Tensor::zeros([1, 1, 3, 1], &Default::default())),
        Some(Tensor::zeros([2, 1, 1, 5], &Default::default())),
    );
    assert_eq!(plane_audit::largest(), 30);
}
