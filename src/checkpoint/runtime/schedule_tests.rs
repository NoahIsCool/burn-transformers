//! Oracle and partition checks of execution grouping.

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;

use super::*;

/// The original map-then-emit grouping, as an independent oracle.
fn reference_groups(classes: &[Option<usize>]) -> Vec<Vec<usize>> {
    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    for (index, class) in classes.iter().enumerate() {
        if let Some(class) = class {
            members.entry(*class).or_default().push(index);
        }
    }
    let mut emitted = BTreeSet::new();
    let mut groups = Vec::new();
    for (index, class) in classes.iter().enumerate() {
        let Some(class) = class else {
            groups.push(vec![index]);
            continue;
        };
        let indices = &members[class];
        if indices.len() == 1 {
            groups.push(vec![index]);
        } else if emitted.insert(*class) {
            groups.push(indices.clone());
        }
    }
    groups
}

/// Fused slices group at their first member; everything else stands alone.
#[test]
fn fused_slices_group_at_their_first_member() {
    let classes = [None, Some(0), Some(1), None, Some(0), Some(2), Some(0)];
    assert_eq!(
        execution_group_indices(&classes, 3),
        vec![vec![0], vec![1, 4, 6], vec![2], vec![3], vec![5]]
    );
    assert!(execution_group_indices(&[], 0).is_empty());
}

proptest! {
    /// Grouping matches the original algorithm and partitions every index once.
    #[test]
    fn groups_match_the_oracle_and_partition_the_plan(
        classes in proptest::collection::vec(proptest::option::of(0_usize..4), 0..24),
    ) {
        let groups = execution_group_indices(&classes, 4);
        prop_assert_eq!(&groups, &reference_groups(&classes));
        let mut seen = vec![0; classes.len()];
        for group in &groups {
            prop_assert!(!group.is_empty());
            prop_assert!(group.windows(2).all(|pair| pair[0] < pair[1]));
            for &index in group {
                seen[index] += 1;
                prop_assert_eq!(classes[index], classes[group[0]]);
            }
            if classes[group[0]].is_none() {
                prop_assert_eq!(group.len(), 1);
            }
        }
        prop_assert!(seen.iter().all(|&count| count == 1));
    }
}
