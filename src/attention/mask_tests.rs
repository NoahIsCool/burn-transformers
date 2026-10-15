//! Exclusion-plane tests: bottom-right causal oracle, unions, and padding planes.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor},
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
