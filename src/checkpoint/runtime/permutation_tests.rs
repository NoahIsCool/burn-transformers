//! Exhaustive checks of probe lengths and axis-permutation recovery.

use super::*;

/// The original used-flag recovery, as an independent oracle.
fn reference_permutation(input: &[usize], output: &[usize]) -> Option<Vec<usize>> {
    if output.len() != input.len() {
        return None;
    }
    let mut axes = vec![0; input.len()];
    let mut used = vec![false; input.len()];
    for (slot, &dimension) in axes.iter_mut().zip(output) {
        let source = input.iter().position(|&candidate| candidate == dimension)?;
        if core::mem::replace(&mut used[source], true) {
            return None;
        }
        *slot = source;
    }
    Some(axes)
}

/// Probe lengths are the tabulated primes, then strictly increasing fallbacks.
#[test]
fn probe_lengths_strictly_increase() {
    let lengths = (0..64).map(probe_dimension).collect::<Vec<_>>();
    assert_eq!(lengths[..8], [2, 3, 5, 7, 11, 13, 17, 19]);
    assert_eq!(lengths[8], 31);
    assert!(lengths.windows(2).all(|pair| pair[0] < pair[1]));
}

/// Every output over the probe's lengths and one stranger matches the oracle.
#[test]
fn recovered_permutations_match_the_used_flag_oracle() {
    let input = (0..4).map(probe_dimension).collect::<Vec<_>>();
    let alphabet = [input[0], input[1], input[2], input[3], 4];
    for rank in 0..=5_u32 {
        for code in 0..alphabet.len().pow(rank) {
            let mut remaining = code;
            let output = (0..rank)
                .map(|_| {
                    let value = alphabet[remaining % alphabet.len()];
                    remaining /= alphabet.len();
                    value
                })
                .collect::<Vec<_>>();
            let mut axes = vec![0; input.len()];
            let found = permutation_of(&input, &output, &mut axes);
            let expected = reference_permutation(&input, &output);
            assert_eq!(found, expected.is_some(), "{output:?}");
            if let Some(expected) = expected {
                assert_eq!(axes, expected, "{output:?}");
                for (k, &source) in axes.iter().enumerate() {
                    assert_eq!(output[k], input[source]);
                }
            }
        }
    }
}
