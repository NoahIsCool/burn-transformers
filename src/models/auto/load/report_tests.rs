//! Generation-default reporting rules.

use proptest::prelude::*;

use super::*;

/// A positive minimum is reported when ignored or lowered, never when zero or met.
#[test]
fn min_new_tokens_report_follows_the_applied_value() {
    assert_eq!(reported_min_new_tokens(None, 0, true), None);
    assert_eq!(reported_min_new_tokens(Some(0), 0, false), None);
    assert_eq!(reported_min_new_tokens(Some(4), 4, true), None);
    assert_eq!(reported_min_new_tokens(Some(300), 256, true), Some(300));
    assert_eq!(reported_min_new_tokens(Some(4), 0, false), Some(4));
}

/// A positive n-gram size is reported exactly when it is not applied.
#[test]
fn no_repeat_ngram_report_is_the_unapplied_case() {
    assert_eq!(reported_no_repeat_ngram(Some(3), true), None);
    assert_eq!(reported_no_repeat_ngram(Some(3), false), Some(3));
    assert_eq!(reported_no_repeat_ngram(Some(0), false), None);
    assert_eq!(reported_no_repeat_ngram(None, false), None);
}

/// Padding accepts at most one token.
#[test]
fn single_token_accepts_at_most_one_id() {
    assert_eq!(single_token(&[]), Ok(None));
    assert_eq!(single_token(&[7]), Ok(Some(7)));
    assert_eq!(single_token(&[7, 8]), Err(MultipleTokens));
}

proptest! {
    /// The verified rules equal the former `Option::filter` chains.
    #[test]
    fn report_rules_match_the_former_filters(
        declared in prop::option::of(0_usize..10),
        applied in 0_usize..10,
        constraints in any::<bool>(),
    ) {
        let positive = declared.filter(|&count| count > 0);
        let former_minimum = if constraints {
            positive.filter(|&count| count > applied)
        } else {
            positive
        };
        prop_assert_eq!(reported_min_new_tokens(declared, applied, constraints), former_minimum);
        let former_ngram = if constraints { None } else { positive };
        prop_assert_eq!(reported_no_repeat_ngram(declared, constraints), former_ngram);
    }

    /// A causal load reports a declared minimum exactly when clamping lowered it.
    #[test]
    fn clamped_minimum_is_reported_exactly_when_lowered(
        declared in 0_usize..20,
        max_new_tokens in 0_usize..20,
    ) {
        let applied = declared.min(max_new_tokens);
        prop_assert_eq!(
            reported_min_new_tokens(Some(declared), applied, true).is_some(),
            declared > max_new_tokens
        );
    }
}
