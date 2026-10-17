//! Connectivity masks against the scalar oracle, Hugging Face conventions, and invalid inputs.

use burn::{
    backend::Flex,
    tensor::{Bool, Tensor, TensorData},
};

use super::*;

/// Deterministic CPU backend for mask construction.
type TestBackend = Flex<f32, i32>;

/// Builds a token-valid mask from host rows.
fn token_mask(rows: &[Vec<bool>]) -> TokenMask<TestBackend> {
    let shape = [rows.len(), rows[0].len()];
    TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_data(
        TensorData::new(rows.concat(), shape),
        &Default::default(),
    ))
}

/// Valid-token ordinal of every column, as the mask builder defines it.
fn ordinals(row: &[bool]) -> Vec<usize> {
    let mut next = 0;
    row.iter()
        .map(|&valid| {
            let ordinal = if valid { next } else { 0 };
            next += usize::from(valid);
            ordinal
        })
        .collect()
}

/// Reference exclusions `[batch, heads, Q, K]` from the scalar oracle.
fn oracle(
    connectivity: AttentionConnectivity,
    causal: bool,
    heads: usize,
    queries: usize,
    key_rows: &[Vec<bool>],
    with_masks: bool,
) -> Vec<bool> {
    oracle_from(
        connectivity,
        causal,
        heads,
        queries,
        key_rows,
        with_masks,
        0,
    )
}

/// Scalar-oracle exclusions whose positions start at absolute `key_start`.
fn oracle_from(
    connectivity: AttentionConnectivity,
    causal: bool,
    heads: usize,
    queries: usize,
    key_rows: &[Vec<bool>],
    with_masks: bool,
    key_start: usize,
) -> Vec<bool> {
    let keys = key_rows[0].len();
    let offset = keys - queries;
    let mut expected = Vec::new();
    for row in key_rows {
        let positions = if with_masks {
            ordinals(row)
        } else {
            (0..keys).collect::<Vec<_>>()
        }
        .into_iter()
        .map(|position| position + key_start)
        .collect::<Vec<_>>();
        let mut plane = Vec::new();
        for query in 0..queries {
            let aligned = query + offset;
            for key in 0..keys {
                let padded = with_masks && (!row[aligned] || !row[key]);
                plane.push(
                    padded || connectivity.excludes(causal, positions[aligned], positions[key]),
                );
            }
        }
        for _ in 0..heads {
            expected.extend_from_slice(&plane);
        }
    }
    expected
}

/// Every rule matches the scalar oracle with and without padding, for suffix and square planes.
#[test]
fn exclusion_masks_match_the_scalar_oracle() {
    let device = Default::default();
    let key_rows = vec![
        vec![true, true, true, true, true, true, true],
        vec![false, false, true, true, false, true, true],
    ];
    for connectivity in [
        AttentionConnectivity::Global,
        AttentionConnectivity::SlidingWindow { window: 1 },
        AttentionConnectivity::SlidingWindow { window: 3 },
        AttentionConnectivity::ChunkedLocal { chunk: 2 },
        AttentionConnectivity::ChunkedLocal { chunk: 3 },
    ] {
        for causal in [false, true] {
            for queries in [1, 3, 7] {
                for with_masks in [false, true] {
                    let keys = key_rows[0].len();
                    let query_rows = key_rows
                        .iter()
                        .map(|row| row[keys - queries..].to_vec())
                        .collect::<Vec<_>>();
                    let key_mask = token_mask(&key_rows);
                    let query_mask = token_mask(&query_rows);
                    let mask = connectivity
                        .exclusion_mask(
                            causal,
                            MaskGeometry::new(2, 2, queries, keys),
                            with_masks.then_some(&query_mask),
                            with_masks.then_some(&key_mask),
                            &device,
                        )
                        .unwrap();
                    let expected = oracle(connectivity, causal, 2, queries, &key_rows, with_masks);
                    let actual = mask.map_or_else(
                        || vec![false; expected.len()],
                        |mask| {
                            assert_eq!(mask.dims(), [2, 2, queries, keys]);
                            mask.into_data().to_vec::<bool>().unwrap()
                        },
                    );
                    assert_eq!(
                        actual, expected,
                        "{connectivity} causal={causal} q={queries} masks={with_masks}"
                    );
                }
            }
        }
    }
}

/// Sliding windows keep `window` positions including the query, as in Hugging Face.
#[test]
fn sliding_window_matches_hugging_face_convention() {
    let window = AttentionConnectivity::SlidingWindow { window: 2 };
    let visible = |query| {
        (0..6)
            .filter(|&key| !window.excludes(true, query, key))
            .collect::<Vec<_>>()
    };
    assert_eq!(visible(0), [0]);
    assert_eq!(visible(3), [2, 3]);
    assert_eq!(visible(5), [4, 5]);
    // Without causality, the window bounds only the past.
    assert!(!window.excludes(false, 1, 5));
    assert!(window.excludes(false, 5, 3));
}

/// Chunked local attention sees exactly its own aligned chunk (Llama 4 iRoPE).
#[test]
fn chunked_local_attention_stays_inside_aligned_chunks() {
    let chunked = AttentionConnectivity::ChunkedLocal { chunk: 4 };
    let visible = |query, causal| {
        (0..10)
            .filter(|&key| !chunked.excludes(causal, query, key))
            .collect::<Vec<_>>()
    };
    assert_eq!(visible(5, true), [4, 5]);
    assert_eq!(visible(5, false), [4, 5, 6, 7]);
    assert_eq!(visible(8, true), [8]);
}

/// A decode step over a cached prefix aligns the single query with the final key.
#[test]
fn cached_decode_step_aligns_with_the_final_key() {
    let mask = AttentionConnectivity::SlidingWindow { window: 3 }
        .exclusion_mask::<TestBackend>(
            true,
            MaskGeometry::new(1, 1, 1, 6),
            None,
            None,
            &Default::default(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        mask.into_data().to_vec::<bool>().unwrap(),
        [true, true, true, false, false, false]
    );
}

/// Padding ordinals skip interior holes so they consume no window capacity.
#[test]
fn ordinals_skip_padding_holes() {
    let mask = token_mask(&[vec![false, true, false, true, true]]);
    assert_eq!(
        valid_token_ordinals(&mask)
            .into_data()
            .to_vec::<i32>()
            .unwrap(),
        [0, 0, 0, 1, 2]
    );
}

/// Global non-causal attention without masks excludes nothing and allocates nothing.
#[test]
fn unrestricted_global_attention_needs_no_mask() {
    let device = Default::default();
    assert!(
        AttentionConnectivity::Global
            .exclusion_mask::<TestBackend>(
                false,
                MaskGeometry::new(2, 3, 4, 5),
                None,
                None,
                &device
            )
            .unwrap()
            .is_none()
    );
    // Global causal attention may have more queries than keys.
    let leading = AttentionConnectivity::Global
        .exclusion_mask::<TestBackend>(true, MaskGeometry::new(1, 1, 3, 2), None, None, &device)
        .unwrap()
        .unwrap();
    assert_eq!(
        leading.into_data().to_vec::<bool>().unwrap(),
        [true, true, false, true, false, false]
    );
}

/// Invalid rules, extents, alignments, and token masks fail with context.
#[test]
fn invalid_requests_fail_with_context() {
    let device = Default::default();
    let geometry = MaskGeometry::new(1, 1, 2, 3);
    for (connectivity, field) in [
        (
            AttentionConnectivity::SlidingWindow { window: 0 },
            "attention.connectivity.window",
        ),
        (
            AttentionConnectivity::ChunkedLocal { chunk: 0 },
            "attention.connectivity.chunk",
        ),
    ] {
        let error = connectivity
            .exclusion_mask::<TestBackend>(true, geometry, None, None, &device)
            .unwrap_err();
        assert!(error.to_string().contains(field), "{error}");
        assert!(connectivity.excludes(false, 0, 0));
    }
    for geometry in [
        MaskGeometry::new(0, 1, 1, 1),
        MaskGeometry::new(1, 0, 1, 1),
        MaskGeometry::new(1, 1, 0, 1),
        MaskGeometry::new(1, 1, 1, 0),
    ] {
        let error = AttentionConnectivity::Global
            .exclusion_mask::<TestBackend>(false, geometry, None, None, &device)
            .unwrap_err();
        assert!(error.to_string().contains("attention.mask_shape"));
    }
    let longer_queries = MaskGeometry::new(1, 1, 3, 2);
    let error = AttentionConnectivity::SlidingWindow { window: 2 }
        .exclusion_mask::<TestBackend>(true, longer_queries, None, None, &device)
        .unwrap_err();
    assert!(error.to_string().contains("attention.connectivity"));
    let keys = token_mask(&[vec![true, true]]);
    let error = AttentionConnectivity::Global
        .exclusion_mask(true, longer_queries, None, Some(&keys), &device)
        .unwrap_err();
    assert!(error.to_string().contains("attention.connectivity"));
    let three = token_mask(&[vec![true, true, true]]);
    let error = AttentionConnectivity::Global
        .exclusion_mask(true, geometry, Some(&three), None, &device)
        .unwrap_err();
    assert!(error.to_string().contains("query_mask"));
    let error = AttentionConnectivity::Global
        .exclusion_mask::<TestBackend>(
            true,
            geometry.with_key_start(usize::MAX - 1),
            None,
            None,
            &device,
        )
        .unwrap_err();
    assert!(error.to_string().contains("attention.mask_shape.key_start"));
}

/// Huge windows and chunks are exact on 32-bit integer backends instead of wrapping.
///
/// Flex converts integer scalars with `as i32`; before reduction, a window of
/// `2^31` excluded every link and `2^32 + 2` behaved like a window of two.
#[test]
fn huge_windows_and_chunks_do_not_wrap_on_32_bit_backends() {
    let device = Default::default();
    let geometry = MaskGeometry::new(1, 1, 4, 6);
    let causal_only = AttentionConnectivity::Global
        .exclusion_mask::<TestBackend>(true, geometry, None, None, &device)
        .unwrap()
        .unwrap()
        .into_data()
        .to_vec::<bool>()
        .unwrap();
    for connectivity in [
        AttentionConnectivity::SlidingWindow { window: 1 << 31 },
        AttentionConnectivity::SlidingWindow {
            window: (1 << 32) + 2,
        },
        AttentionConnectivity::SlidingWindow { window: usize::MAX },
        AttentionConnectivity::ChunkedLocal {
            chunk: (1 << 32) + 2,
        },
        AttentionConnectivity::ChunkedLocal { chunk: usize::MAX },
    ] {
        let mask = connectivity
            .exclusion_mask::<TestBackend>(true, geometry, None, None, &device)
            .unwrap()
            .unwrap()
            .into_data()
            .to_vec::<bool>()
            .unwrap();
        assert_eq!(mask, causal_only, "{connectivity}");
        // Without causality the rule alone excludes nothing.
        assert!(
            connectivity
                .exclusion_mask::<TestBackend>(false, geometry, None, None, &device)
                .unwrap()
                .is_none(),
            "{connectivity}"
        );
    }
    // A huge chunk whose boundary falls inside the keys still splits them exactly.
    let chunk = (1usize << 32) + 2;
    let straddling = geometry.with_key_start(chunk - 3);
    let mask = AttentionConnectivity::ChunkedLocal { chunk }
        .exclusion_mask::<TestBackend>(false, straddling, None, None, &device)
        .unwrap()
        .unwrap()
        .into_data()
        .to_vec::<bool>()
        .unwrap();
    let rows = [
        [false, false, false, true, true, true],
        [true, true, true, false, false, false],
    ];
    // Query rows 0..4 sit at local key positions 2..6: row 0 is before the boundary.
    let expected = [rows[0], rows[1], rows[1], rows[1]].concat();
    assert_eq!(mask, expected);
}

/// Position arithmetic beyond the device's `i32` range is rejected before allocation.
#[test]
fn positions_beyond_the_integer_dtype_are_rejected() {
    let device = Default::default();
    let too_long = MaskGeometry::new(1, 1, 1, (1 << 31) + 1);
    let error = AttentionConnectivity::Global
        .exclusion_mask::<TestBackend>(true, too_long, None, None, &device)
        .unwrap_err();
    assert!(error.to_string().contains("attention.positions"), "{error}");
    // `key + window` reaches `K - 1 + window`, which must also fit.
    let wide = MaskGeometry::new(1, 1, 1, (1 << 30) + 2);
    let error = AttentionConnectivity::SlidingWindow {
        window: (1 << 30) + 1,
    }
    .exclusion_mask::<TestBackend>(true, wide, None, None, &device)
    .unwrap_err();
    assert!(error.to_string().contains("I32"), "{error}");
}

/// Chunk boundaries follow absolute positions when keys start after evicted tokens.
#[test]
fn chunked_local_honors_absolute_key_start() {
    let device = Default::default();
    let key_rows = vec![
        vec![true, true, true, true, true, true, true],
        vec![false, true, true, false, true, true, true],
    ];
    for chunk in [2, 3, 4, 9] {
        for key_start in [0, 1, 2, 5, 11] {
            for causal in [false, true] {
                for with_masks in [false, true] {
                    let connectivity = AttentionConnectivity::ChunkedLocal { chunk };
                    let queries = 3;
                    let query_rows = key_rows
                        .iter()
                        .map(|row| row[4..].to_vec())
                        .collect::<Vec<_>>();
                    let key_mask = token_mask(&key_rows);
                    let query_mask = token_mask(&query_rows);
                    let actual = connectivity
                        .exclusion_mask(
                            causal,
                            MaskGeometry::new(2, 1, queries, 7).with_key_start(key_start),
                            with_masks.then_some(&query_mask),
                            with_masks.then_some(&key_mask),
                            &device,
                        )
                        .unwrap()
                        .map_or_else(
                            || vec![false; 2 * queries * 7],
                            |mask| mask.into_data().to_vec::<bool>().unwrap(),
                        );
                    let expected = oracle_from(
                        connectivity,
                        causal,
                        1,
                        queries,
                        &key_rows,
                        with_masks,
                        key_start,
                    );
                    assert_eq!(
                        actual, expected,
                        "C={chunk} start={key_start} causal={causal} masks={with_masks}"
                    );
                }
            }
        }
    }
}

/// Validating token masks builds no `[query, key]` plane; chunks build only their own.
#[test]
fn exclusion_plans_build_planes_only_per_chunk() {
    use super::super::mask::plane_audit;

    let device = Default::default();
    let rows = vec![vec![true; 64], [vec![false; 3], vec![true; 61]].concat()];
    let mask = token_mask(&rows);
    plane_audit::reset();
    let plan = ExclusionPlan::new(
        AttentionConnectivity::SlidingWindow { window: 8 },
        true,
        MaskGeometry::new(2, 4, 64, 64),
        Some(&mask),
        Some(&mask),
        &device,
    )
    .unwrap();
    assert_eq!(plane_audit::largest(), 0, "validation built a plane");
    plan.chunk(8..16, 0..64, 4).unwrap().unwrap();
    assert_eq!(plane_audit::largest(), 2 * 8 * 64);
}

/// Rules render compactly and round-trip through serde with snake-case tags.
#[test]
fn connectivity_displays_and_serializes() {
    let rules = [
        AttentionConnectivity::Global,
        AttentionConnectivity::SlidingWindow { window: 4096 },
        AttentionConnectivity::ChunkedLocal { chunk: 8192 },
    ];
    let names = rules.map(|rule| rule.to_string());
    assert_eq!(
        names,
        ["global", "sliding_window(4096)", "chunked_local(8192)"]
    );
    for rule in rules {
        let json = serde_json::to_string(&rule).unwrap();
        assert_eq!(
            serde_json::from_str::<AttentionConnectivity>(&json).unwrap(),
            rule
        );
    }
    assert_eq!(
        serde_json::to_string(&AttentionConnectivity::SlidingWindow { window: 2 }).unwrap(),
        r#"{"sliding_window":{"window":2}}"#
    );
    assert_eq!(
        AttentionConnectivity::default(),
        AttentionConnectivity::Global
    );
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(64))]

    /// Random padding patterns, lengths, and key offsets agree with the scalar oracle.
    #[test]
    fn random_masks_match_the_oracle(
        rule in 0usize..3,
        size in 1usize..=4,
        causal in proptest::bool::ANY,
        queries in 1usize..=5,
        extra_keys in 0usize..=3,
        valid_bits in proptest::collection::vec(proptest::bool::ANY, 16),
        key_start in 0usize..=9,
    ) {
        let connectivity = match rule {
            0 => AttentionConnectivity::Global,
            1 => AttentionConnectivity::SlidingWindow { window: size },
            _ => AttentionConnectivity::ChunkedLocal { chunk: size },
        };
        let keys = queries + extra_keys;
        let key_rows = vec![valid_bits[..keys].to_vec()];
        let query_rows = vec![key_rows[0][keys - queries..].to_vec()];
        let mask = connectivity.exclusion_mask(
            causal,
            MaskGeometry::new(1, 1, queries, keys).with_key_start(key_start),
            Some(&token_mask(&query_rows)),
            Some(&token_mask(&key_rows)),
            &Default::default(),
        ).unwrap();
        let expected = oracle_from(connectivity, causal, 1, queries, &key_rows, true, key_start);
        let actual = mask.map_or_else(|| vec![false; expected.len()], |mask| mask.into_data().to_vec::<bool>().unwrap());
        proptest::prop_assert_eq!(actual, expected);
    }
}

/// Lengths beyond `i64` and windows whose positions overflow `i64` fail before any allocation.
///
/// Positions are signed integers, so a query or key axis longer than
/// `i64::MAX` is rejected by geometry validation, and a window that survives
/// reduction (it is shorter than the keys) but whose largest formed position
/// `K - 1 + window` exceeds `i64::MAX` is rejected by the peak check. Every
/// geometry here is far too large to allocate, so reaching a tensor builder
/// would abort the test.
#[test]
fn position_ranges_beyond_i64_are_rejected_from_metadata() {
    let device = Default::default();
    let beyond = 1_usize << 63;
    for (geometry, reason) in [
        (
            MaskGeometry::new(1, 1, beyond, beyond),
            "query length exceeds i64",
        ),
        (MaskGeometry::new(1, 1, 1, beyond), "key length exceeds i64"),
    ] {
        let error = AttentionConnectivity::Global
            .exclusion_mask::<TestBackend>(true, geometry, None, None, &device)
            .unwrap_err();
        assert!(
            matches!(&error, Error::Configuration { field, reason: actual }
                if field == "attention.causal_mask" && actual == reason),
            "{error:?}"
        );
    }
    let keys = (1_usize << 62) + 2;
    let error = AttentionConnectivity::SlidingWindow { window: keys - 1 }
        .exclusion_mask::<TestBackend>(true, MaskGeometry::new(1, 1, 1, keys), None, None, &device)
        .unwrap_err();
    assert!(
        matches!(&error, Error::Configuration { field, reason }
            if field == "attention.positions" && reason == "position arithmetic exceeds i64"),
        "{error:?}"
    );
}
