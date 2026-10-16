//! Executable checks of the Verus-verified head-grouping arithmetic against brute force.

use super::*;

/// Group counts exist exactly for non-zero, divisible head counts.
#[test]
fn head_groups_require_exact_division() {
    assert_eq!(head_groups(8, 2), Some(4));
    assert_eq!(head_groups(8, 8), Some(1));
    assert_eq!(head_groups(8, 3), None);
    assert_eq!(head_groups(0, 2), None);
    assert_eq!(head_groups(8, 0), None);
    assert_eq!(head_groups(0, 0), None);
    for queries in 0..=12usize {
        for keys in 0..=12usize {
            let expected = (queries > 0 && keys > 0 && queries % keys == 0).then(|| queries / keys);
            assert_eq!(head_groups(queries, keys), expected, "{queries} / {keys}");
        }
    }
}

/// Repetition by expand-and-reshape places repeated head `m` at expanded `[m / g, m mod g]`.
#[test]
fn repetition_reads_source_head_of_each_group() {
    for key_value_heads in 1..=4usize {
        for groups in 1..=4usize {
            let query_heads = key_value_heads * groups;
            assert_eq!(head_groups(query_heads, key_value_heads), Some(groups));
            for rows in 1..=4usize {
                // Every repeated element is a distinct expanded element.
                let mut seen = vec![false; 2 * query_heads * rows];
                for batch in 0..2 {
                    for head in 0..query_heads {
                        for row in 0..rows {
                            let flat = (batch * query_heads + head) * rows + row;
                            let expanded = ((batch * key_value_heads + head / groups) * groups
                                + head % groups)
                                * rows
                                + row;
                            assert_eq!(flat, expanded, "G={key_value_heads} g={groups}");
                            assert!(!seen[expanded]);
                            seen[expanded] = true;
                        }
                    }
                }
            }
        }
    }
}

/// Folding is a contiguous view: flat indices survive and every folded row has one origin.
#[test]
fn folding_preserves_flat_indices_and_inverts() {
    for key_value_heads in 1..=4usize {
        for groups in 1..=4usize {
            let query_heads = key_value_heads * groups;
            for rows in 1..=5usize {
                let folded = folded_rows(query_heads, key_value_heads, rows);
                assert_eq!(folded, groups * rows);
                assert_eq!(unfolded_rows(query_heads, key_value_heads, folded), rows);
                let mut seen = vec![false; key_value_heads * folded];
                for batch in 0..2 {
                    for head in 0..query_heads {
                        for row in 0..rows {
                            let kv_head = head / groups;
                            let folded_row = (head % groups) * rows + row;
                            assert!(kv_head < key_value_heads && folded_row < folded);
                            assert_eq!(
                                (batch * query_heads + head) * rows + row,
                                (batch * key_value_heads + kv_head) * folded + folded_row
                            );
                            if batch == 0 {
                                assert!(!seen[kv_head * folded + folded_row]);
                                seen[kv_head * folded + folded_row] = true;
                                // The sink selector's member index is the group member.
                                assert_eq!(folded_row / rows, head % groups);
                            }
                        }
                    }
                }
                assert!(seen.into_iter().all(|hit| hit));
            }
        }
    }
}

/// Unfolding truncates like integer division when rows are not a whole multiple.
#[test]
fn unfolding_divides_by_the_group_size() {
    assert_eq!(unfolded_rows(8, 2, 12), 3);
    assert_eq!(unfolded_rows(8, 2, 13), 3);
    assert_eq!(unfolded_rows(4, 4, 7), 7);
}

/// The source value heads each query head reads after the planned construction.
///
/// Repeated plans pair repeated heads `(h mod half, h mod half + half)`;
/// `repeat_key_value` makes repeated head `m` a copy of source head `m / g`.
fn built_pairs(query_heads: usize, key_value_heads: usize) -> Vec<(usize, usize)> {
    let groups = query_heads / key_value_heads;
    match value_pairing(query_heads, key_value_heads) {
        ValuePairing::Shared => vec![(0, 0); query_heads],
        ValuePairing::Native { half } => (0..query_heads)
            .map(|head| {
                let kv = head / groups;
                (kv % half, kv % half + half)
            })
            .collect(),
        ValuePairing::Repeated { half } => (0..query_heads)
            .map(|head| {
                let (first, second) = (head % half, head % half + half);
                assert!(second < query_heads, "repeated head out of range");
                (first / groups, second / groups)
            })
            .collect(),
    }
}

/// Every query head reads the reference pair of its head index modulo `H / 2`.
#[test]
fn value_pairing_reproduces_reference_pairs() {
    for key_value_heads in 1..=6usize {
        for groups in 1..=6usize {
            let query_heads = key_value_heads * groups;
            if query_heads % 2 != 0 {
                continue;
            }
            let pairs = query_heads / 2;
            let plan = value_pairing(query_heads, key_value_heads);
            assert_eq!(plan == ValuePairing::Shared, key_value_heads == 1);
            match plan {
                ValuePairing::Native { half } => {
                    assert!(key_value_heads % 2 == 0 && half * 2 == key_value_heads);
                }
                ValuePairing::Repeated { half } => {
                    assert!(key_value_heads % 2 == 1 && half * 2 == query_heads);
                }
                ValuePairing::Shared => {}
            }
            let built = built_pairs(query_heads, key_value_heads);
            for (head, pair) in built.into_iter().enumerate() {
                let low = head % pairs;
                let reference = (low / groups, (low + pairs) / groups);
                assert_eq!(
                    pair, reference,
                    "H={query_heads} KV={key_value_heads} h={head}"
                );
            }
        }
    }
}
