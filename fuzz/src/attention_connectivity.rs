//! Attention connectivity masks and masked grouped-query attention against scalar oracles.
//!
//! [`AttentionConnectivity::exclusion_mask`] builds the padding, causal,
//! sliding-window, and chunked-local exclusions of one attention call
//! (`true == excluded`). Queries are the bottom-right suffix of the keys; with
//! a key mask, positions are valid-token ordinals, so padding consumes no
//! window; [`MaskGeometry::key_start`] shifts absolute positions, which only
//! chunk boundaries observe. For small random geometries, rules, and token
//! masks (including fully padded rows) the harness checks:
//!
//! 1. **Acceptance.** Building fails exactly for a zero window or chunk, an
//!    empty axis, `key_start + K` overflowing `usize`, a mis-shaped token mask,
//!    or more queries than keys outside unpadded global attention.
//! 2. **Mask values.** Every head of every row equals [`expected_excluded`],
//!    an independent scalar statement of the rules over `u128` positions.
//! 3. **Window key ranges.** For causal sliding windows,
//!    [`sliding_window_key_range`] of every query-row range is exactly the union
//!    of keys those rows may see.
//! 4. **Attention.** [`grouped_query_attention`] with the mask equals an `f64`
//!    softmax over the visible keys (zero output for fully excluded rows), in
//!    both precision modes, with native KV heads (`H_kv` divides `H`); passing
//!    causality as the backend's `is_causal` flag instead of in the mask gives
//!    the same result (bottom-right alignment on every backend).

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, TensorData, ops::AttentionModuleOptions},
};
use transformers::{
    TokenMask,
    attention::{
        AttentionConnectivity, AttentionPrecisionMode, MaskGeometry, grouped_query_attention,
        sliding_window_key_range,
    },
};

use crate::Bytes;

/// Host backend of every tensor in this harness.
type Backend = Flex;

/// Largest query or key length the harness builds.
const MAX_SEQUENCE: usize = 8;
/// Tolerance between `f32` attention and the `f64` oracle.
const TOLERANCE: f64 = 2e-5;

/// One decoded mask problem.
#[derive(Debug, Clone)]
struct Problem {
    /// Visibility rule.
    connectivity: AttentionConnectivity,
    /// Whether keys after the query position are excluded.
    causal: bool,
    /// Extents and absolute key start.
    geometry: MaskGeometry,
    /// Optional `[batch, Q]` query validity, possibly mis-shaped.
    query_valid: Option<(usize, usize, Vec<bool>)>,
    /// Optional `[batch, K]` key validity, possibly mis-shaped.
    key_valid: Option<(usize, usize, Vec<bool>)>,
}

impl Problem {
    /// Whether the mask must build, per the documented rules.
    fn accepts(&self) -> bool {
        let MaskGeometry {
            batch,
            heads,
            query_sequence,
            key_sequence,
            key_start,
        } = self.geometry;
        let rule_valid = !matches!(
            self.connectivity,
            AttentionConnectivity::SlidingWindow { window: 0 }
                | AttentionConnectivity::ChunkedLocal { chunk: 0 }
        );
        let axes = batch > 0 && heads > 0 && query_sequence > 0 && key_sequence > 0;
        let shaped = |mask: &Option<(usize, usize, Vec<bool>)>, length: usize| {
            mask.as_ref()
                .is_none_or(|(rows, columns, _)| *rows == batch && *columns == length)
        };
        let global = self.connectivity == AttentionConnectivity::Global;
        let alignment = key_sequence >= query_sequence
            || (global && !(self.causal && self.key_valid.is_some()));
        rule_valid
            && axes
            && key_start.checked_add(key_sequence).is_some()
            && shaped(&self.query_valid, query_sequence)
            && shaped(&self.key_valid, key_sequence)
            && alignment
    }
}

/// Builds a token mask from host validity.
fn token_mask(rows: usize, columns: usize, valid: &[bool]) -> TokenMask<Backend> {
    let values: Vec<i32> = valid.iter().map(|&valid| i32::from(valid)).collect();
    TokenMask::new(
        Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(values, [rows, columns]),
            &Default::default(),
        )
        .bool(),
    )
}

/// Scalar oracle: whether query row `q` of batch row `b` must not see key `k`.
///
/// Positions are `u128` so `key_start` near `usize::MAX` cannot overflow. With a
/// key mask, a valid key's position is the number of valid keys before it
/// (padding gets zero), and a query takes the position of its key column
/// `K - Q + q`; otherwise positions are column indices. Both are offset by
/// `key_start`.
fn expected_excluded(problem: &Problem, b: usize, q: usize, k: usize) -> bool {
    let MaskGeometry {
        query_sequence,
        key_sequence,
        key_start,
        ..
    } = problem.geometry;
    if let Some((_, _, valid)) = &problem.query_valid
        && !valid[b * query_sequence + q]
    {
        return true;
    }
    let start = key_start as i128;
    let key_valid = |column: usize| {
        problem
            .key_valid
            .as_ref()
            .is_none_or(|(_, _, valid)| valid[b * key_sequence + column])
    };
    if !key_valid(k) {
        return true;
    }
    if !problem.causal && problem.connectivity == AttentionConnectivity::Global {
        // Only padding applies; positions (possibly negative here) are unused.
        return false;
    }
    let (query_position, key_position) = match &problem.key_valid {
        Some((_, _, valid)) => {
            let ordinal = |column: usize| -> i128 {
                if valid[b * key_sequence + column] {
                    let row = &valid[b * key_sequence..b * key_sequence + column];
                    row.iter().filter(|&&valid| valid).count() as i128
                } else {
                    0
                }
            };
            let column = key_sequence - query_sequence + q;
            (start + ordinal(column), start + ordinal(k))
        }
        None => (
            start + q as i128 + key_sequence as i128 - query_sequence as i128,
            start + k as i128,
        ),
    };
    if problem.causal && key_position > query_position {
        return true;
    }
    match problem.connectivity {
        AttentionConnectivity::Global => false,
        AttentionConnectivity::SlidingWindow { window } => {
            query_position - key_position >= window as i128
        }
        AttentionConnectivity::ChunkedLocal { chunk } => {
            query_position.div_euclid(chunk as i128) != key_position.div_euclid(chunk as i128)
        }
    }
}

/// Decodes a validity mask that is usually well shaped.
fn validity(
    bytes: &mut Bytes<'_>,
    batch: usize,
    length: usize,
) -> Option<(usize, usize, Vec<bool>)> {
    if !bytes.flag() {
        return None;
    }
    let (rows, columns) = match bytes.byte() {
        250 => (batch + 1, length),
        251 => (batch, length + 1),
        _ => (batch, length),
    };
    // Bias toward valid tokens, with whole rows sometimes padded out.
    let fully_padded_row = bytes.below(4 * rows.max(1));
    let valid = (0..rows * columns)
        .map(|index| index / columns.max(1) != fully_padded_row && !bytes.byte().is_multiple_of(4))
        .collect();
    Some((rows, columns, valid))
}

/// Decodes one problem; see [`exercise`] for the byte layout.
fn decode(bytes: &mut Bytes<'_>) -> Problem {
    let parameter = |bytes: &mut Bytes<'_>| match bytes.byte() {
        250 => 0,
        251 => usize::MAX,
        252 => MAX_SEQUENCE,
        value => usize::from(value % 9) + 1,
    };
    let connectivity = match bytes.below(3) {
        0 => AttentionConnectivity::Global,
        1 => AttentionConnectivity::SlidingWindow {
            window: parameter(bytes),
        },
        _ => AttentionConnectivity::ChunkedLocal {
            chunk: parameter(bytes),
        },
    };
    let causal = bytes.flag();
    let extent = |bytes: &mut Bytes<'_>, bound: usize| match bytes.byte() {
        255 => 0,
        value => usize::from(value) % bound + 1,
    };
    let batch = extent(bytes, 3);
    let heads = extent(bytes, 3);
    let query_sequence = extent(bytes, MAX_SEQUENCE);
    let key_sequence = extent(bytes, MAX_SEQUENCE);
    let key_start = bytes.pick(&[0, 0, 1, 3, 5, 7, usize::MAX - 8, usize::MAX - 4, usize::MAX]);
    let geometry =
        MaskGeometry::new(batch, heads, query_sequence, key_sequence).with_key_start(key_start);
    let query_valid = validity(bytes, batch, query_sequence);
    let key_valid = validity(bytes, batch, key_sequence);
    Problem {
        connectivity,
        causal,
        geometry,
        query_valid,
        key_valid,
    }
}

/// Builds the mask of `problem`, or `None` on a documented rejection.
fn build(problem: &Problem) -> Option<Option<Tensor<Backend, 4, Bool>>> {
    let query = problem
        .query_valid
        .as_ref()
        .map(|(rows, columns, valid)| token_mask(*rows, *columns, valid));
    let key = problem
        .key_valid
        .as_ref()
        .map(|(rows, columns, valid)| token_mask(*rows, *columns, valid));
    problem
        .connectivity
        .exclusion_mask::<Backend>(
            problem.causal,
            problem.geometry,
            query.as_ref(),
            key.as_ref(),
            &Default::default(),
        )
        .ok()
}

/// Reads a mask into host values, materializing `None` as all-visible.
fn mask_values(problem: &Problem, mask: Option<Tensor<Backend, 4, Bool>>) -> Vec<bool> {
    let MaskGeometry {
        batch,
        heads,
        query_sequence,
        key_sequence,
        ..
    } = problem.geometry;
    match mask {
        Some(mask) => {
            assert_eq!(mask.dims(), [batch, heads, query_sequence, key_sequence]);
            mask.into_data().to_vec::<bool>().expect("Boolean mask")
        }
        None => vec![false; batch * heads * query_sequence * key_sequence],
    }
}

/// Checks the mask of an accepted problem against the scalar oracle.
fn check_mask(problem: &Problem, values: &[bool]) {
    let MaskGeometry {
        batch,
        heads,
        query_sequence,
        key_sequence,
        ..
    } = problem.geometry;
    let mut index = 0;
    for b in 0..batch {
        for _head in 0..heads {
            for q in 0..query_sequence {
                for k in 0..key_sequence {
                    assert_eq!(
                        values[index],
                        expected_excluded(problem, b, q, k),
                        "row {b} query {q} key {k} of {problem:?}"
                    );
                    index += 1;
                }
            }
        }
    }
}

/// Checks every row range's sliding-window key range against the visible keys.
fn check_window_ranges(problem: &Problem) {
    let MaskGeometry {
        query_sequence,
        key_sequence,
        ..
    } = problem.geometry;
    let AttentionConnectivity::SlidingWindow { window } = problem.connectivity else {
        return;
    };
    for start in 0..=query_sequence {
        for end in start..=query_sequence + 1 {
            let range = sliding_window_key_range(query_sequence, key_sequence, start..end, window);
            let valid = start < end
                && end <= query_sequence
                && query_sequence <= key_sequence
                && window > 0;
            assert_eq!(range.is_some(), valid);
            let Some(range) = range else { continue };
            let offset = key_sequence - query_sequence;
            let visible: Vec<usize> = (0..key_sequence)
                .filter(|&k| {
                    (start..end).any(|q| {
                        let position = q + offset;
                        k <= position && position - k < window
                    })
                })
                .collect();
            assert_eq!(
                Some(range.start..range.end),
                visible
                    .first()
                    .map(|&first| first..visible.last().unwrap() + 1),
                "rows {start}..{end}"
            );
        }
    }
}

/// Deterministic small values on a quarter grid from a byte stream.
fn values(bytes: &mut Bytes<'_>, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| (f32::from(bytes.byte() % 17) - 8.0) / 4.0)
        .collect()
}

/// Checks grouped-query attention under the mask against an `f64` softmax.
fn check_attention(bytes: &mut Bytes<'_>, problem: &Problem, excluded: &[bool]) {
    let MaskGeometry {
        batch,
        heads,
        query_sequence,
        key_sequence,
        ..
    } = problem.geometry;
    let kv_heads = if heads > 1 && bytes.flag() { 1 } else { heads };
    let width = bytes.below(3) + 1;
    let device = Default::default();
    let query = values(bytes, batch * heads * query_sequence * width);
    let key = values(bytes, batch * kv_heads * key_sequence * width);
    let value = values(bytes, batch * kv_heads * key_sequence * width);
    let tensor = |data: &[f32], heads: usize, length: usize| {
        Tensor::<Backend, 4>::from_data(
            TensorData::new(data.to_vec(), [batch, heads, length, width]),
            &device,
        )
    };
    let mask = Tensor::<Backend, 4, Int>::from_data(
        TensorData::new(
            excluded.iter().map(|&e| i32::from(e)).collect::<Vec<_>>(),
            [batch, heads, query_sequence, key_sequence],
        ),
        &device,
    )
    .bool();
    // Expected output, one f64 softmax per (row, head, query).
    let group = heads / kv_heads;
    let scale = 1.0 / (width as f64).sqrt();
    let mut expected = Vec::with_capacity(batch * heads * query_sequence * width);
    for b in 0..batch {
        for h in 0..heads {
            let kv = h / group;
            for q in 0..query_sequence {
                let query_row = &query[((b * heads + h) * query_sequence + q) * width..][..width];
                let mut weights = Vec::with_capacity(key_sequence);
                for k in 0..key_sequence {
                    let mask_index = ((b * heads + h) * query_sequence + q) * key_sequence + k;
                    if excluded[mask_index] {
                        weights.push(None);
                        continue;
                    }
                    let key_row = &key[((b * kv_heads + kv) * key_sequence + k) * width..][..width];
                    let score: f64 = query_row
                        .iter()
                        .zip(key_row)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    weights.push(Some(score * scale));
                }
                let maximum = weights
                    .iter()
                    .flatten()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = weights
                    .iter()
                    .map(|weight| weight.map_or(0.0, |w| (w - maximum).exp()))
                    .collect();
                let total: f64 = exps.iter().sum();
                for channel in 0..width {
                    let mut sum = 0.0;
                    if total > 0.0 {
                        for (k, &e) in exps.iter().enumerate() {
                            let row = ((b * kv_heads + kv) * key_sequence + k) * width;
                            sum += e / total * f64::from(value[row + channel]);
                        }
                    }
                    expected.push(sum);
                }
            }
        }
    }
    let unmasked_causal = (problem.causal
        && problem.connectivity == AttentionConnectivity::Global
        && problem.key_valid.is_none()
        && problem.query_valid.is_none())
    .then(|| {
        // Causality from the backend flag instead of the mask.
        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: true,
        };
        (None, options)
    });
    let masked = (
        Some(mask),
        AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        },
    );
    for (mask, options) in [Some(masked), unmasked_causal].into_iter().flatten() {
        for mode in [
            AttentionPrecisionMode::BackendDefault,
            AttentionPrecisionMode::EagerFp32Softmax,
        ] {
            let output = grouped_query_attention(
                tensor(&query, heads, query_sequence),
                tensor(&key, kv_heads, key_sequence),
                tensor(&value, kv_heads, key_sequence),
                mask.clone(),
                None,
                options,
                mode,
            )
            .expect("valid attention operands");
            assert_eq!(output.dims(), [batch, heads, query_sequence, width]);
            let actual = output.into_data().to_vec::<f32>().expect("F32 output");
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (f64::from(actual) - expected).abs() <= TOLERANCE,
                    "{mode:?} causal-flag {}: element {index} is {actual}, expected {expected}",
                    options.is_causal
                );
            }
        }
    }
}

/// Decodes and checks one connectivity problem.
///
/// Byte layout: rule kind (global, window, chunk) and its parameter (250 is
/// zero, 251 `usize::MAX`), causality, batch, heads, query and key lengths (255
/// is zero), the key start (table index), then optional query and key validity
/// masks (a flag, a shape byte where 250/251 mis-shape it, a fully padded row
/// selector, and one byte per token), and finally the attention operands: a
/// KV-head choice, the head width (1..=3), and query, key, and value values.
/// Returns whether the mask was built, so tests can prove both outcomes occur.
pub fn exercise(input: &[u8]) -> bool {
    let mut bytes = Bytes::new(input);
    let problem = decode(&mut bytes);
    let built = build(&problem);
    assert_eq!(built.is_some(), problem.accepts(), "{problem:?}");
    check_window_ranges(&problem);
    let Some(mask) = built else { return false };
    let values = mask_values(&problem, mask);
    check_mask(&problem, &values);
    check_attention(&mut bytes, &problem, &values);
    true
}

#[cfg(test)]
#[path = "attention_connectivity_tests.rs"]
mod tests;
