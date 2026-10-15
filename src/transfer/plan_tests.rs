//! Transfer precedence and tie bookkeeping tests against direct reference definitions.

use super::*;

/// Shorthands for rule classifications.
const U: RuleMatch = RuleMatch::Unrelated;
/// An exact copy or transform.
const E: RuleMatch = RuleMatch::Exact;
/// A prefix or matching-path rule.
const F: RuleMatch = RuleMatch::Fallback;

/// Kept targets, the first exact rule, then a unique fallback decide the source.
#[test]
fn rule_precedence_is_keep_exact_then_unique_fallback() {
    assert_eq!(select_rule(true, &[E, F]), RuleSelection::Unselected);
    assert_eq!(select_rule(false, &[]), RuleSelection::Unselected);
    assert_eq!(select_rule(false, &[U, U]), RuleSelection::Unselected);
    assert_eq!(select_rule(false, &[F, E, E]), RuleSelection::Rule(1));
    assert_eq!(select_rule(false, &[U, F, U]), RuleSelection::Rule(1));
    assert_eq!(select_rule(false, &[F, U, F]), RuleSelection::Overlapping);
    assert_eq!(select_rule(false, &[F, F, E]), RuleSelection::Rule(2));
}

/// Builds a selected tie member.
fn key(origin: u64, transform: Option<usize>, trainable: bool) -> Option<StagedKey> {
    Some(StagedKey {
        origin,
        transform,
        trainable,
    })
}

/// Target ties are untouched, untied, consistent, or inconsistent.
#[test]
fn target_ties_need_one_shared_selection() {
    assert_eq!(target_tie(&[None, None], false), TargetTie::Untouched);
    assert_eq!(target_tie(&[None, None], true), TargetTie::Untouched);
    assert_eq!(
        target_tie(&[None, key(1, None, true)], true),
        TargetTie::Untied
    );
    assert_eq!(
        target_tie(&[key(1, None, true), key(1, None, true)], false),
        TargetTie::Consistent
    );
    assert_eq!(
        target_tie(&[None, key(1, None, true)], false),
        TargetTie::Inconsistent
    );
    assert_eq!(
        target_tie(&[key(1, Some(0), true), key(1, None, true)], false),
        TargetTie::Inconsistent
    );
    assert_eq!(
        target_tie(&[key(1, None, true), key(2, None, true)], false),
        TargetTie::Inconsistent
    );
    assert_eq!(
        target_tie(&[key(1, None, true), key(1, None, false)], false),
        TargetTie::Inconsistent
    );
}

/// The first path that reaches two target classes is reported.
#[test]
fn fanned_out_paths_are_found_in_path_order() {
    assert_eq!(first_fanned_out_path(&[], &[], 0), None);
    assert_eq!(first_fanned_out_path(&[0, 1, 0], &[7, 8, 7], 2), None);
    assert_eq!(
        first_fanned_out_path(&[1, 0, 1, 0], &[7, 8, 9, 8], 2),
        Some(1)
    );
    assert_eq!(
        first_fanned_out_path(&[1, 0, 1, 0], &[7, 8, 9, 6], 2),
        Some(0)
    );
}

proptest::proptest! {
    /// Precedence matches the direct definition of the original iterator implementation.
    #[test]
    fn select_rule_matches_its_definition(
        kept in proptest::bool::ANY,
        raw in proptest::collection::vec(0u8..3, 0..8),
    ) {
        let matches = raw
            .iter()
            .map(|kind| [U, E, F][usize::from(*kind)])
            .collect::<Vec<_>>();
        let expected = if kept {
            RuleSelection::Unselected
        } else if let Some(index) = matches.iter().position(|kind| *kind == E) {
            RuleSelection::Rule(index)
        } else {
            let fallbacks = matches
                .iter()
                .enumerate()
                .filter(|(_, kind)| **kind == F)
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            match fallbacks.as_slice() {
                [] => RuleSelection::Unselected,
                [index] => RuleSelection::Rule(*index),
                _ => RuleSelection::Overlapping,
            }
        };
        proptest::prop_assert_eq!(select_rule(kept, &matches), expected);
    }

    /// Fan-out detection matches a grouped reference.
    #[test]
    fn fanned_out_paths_match_a_grouped_reference(
        raw in proptest::collection::vec((0usize..4, 0u64..3), 0..10),
    ) {
        let paths = raw.iter().map(|(path, _)| *path).collect::<Vec<_>>();
        let classes = raw.iter().map(|(_, class)| *class).collect::<Vec<_>>();
        let expected = (0..4).find(|path| {
            let reached = raw
                .iter()
                .filter(|(candidate, _)| candidate == path)
                .map(|(_, class)| *class)
                .collect::<std::collections::BTreeSet<_>>();
            reached.len() > 1
        });
        proptest::prop_assert_eq!(first_fanned_out_path(&paths, &classes, 4), expected);
    }
}
