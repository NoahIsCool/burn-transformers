//! Tests for special-token ID declarations and model-vocabulary bounds.

use proptest::prelude::*;

use super::*;

/// Declared IDs fit `u32`, agree with the vocabulary, and index the embedding table.
#[test]
fn special_ids_follow_the_vocabulary() {
    assert_eq!(declared_special_id(7), Some(7));
    assert_eq!(declared_special_id(u64::from(u32::MAX)), Some(u32::MAX));
    assert_eq!(declared_special_id(u64::from(u32::MAX) + 1), None);
    assert_eq!(reconcile_special_id(None, 3), Ok(3));
    assert_eq!(reconcile_special_id(Some(3), 3), Ok(3));
    assert_eq!(reconcile_special_id(Some(4), 3), Err(4));
    assert!(id_fits_vocabulary(9, 10));
    assert!(!id_fits_vocabulary(10, 10));
    assert!(!id_fits_vocabulary(0, 0));
}

proptest! {
    /// Declared IDs agree with `u32::try_from`.
    #[test]
    fn declared_ids_match_try_from(value in any::<u64>()) {
        prop_assert_eq!(declared_special_id(value), u32::try_from(value).ok());
    }
}
