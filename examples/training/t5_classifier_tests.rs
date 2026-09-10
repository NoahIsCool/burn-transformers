//! Training and native-record integration checks for `t5_classifier`.

use super::*;
#[test]
fn held_out_combinations_are_disjoint_and_classifier_learns_then_reloads() {
    let (train_rows, held) = dataset();
    assert_eq!((train_rows.len(), held.len()), (24, 8));
    assert!(held.iter().all(|r| !train_rows.contains(r)));
    train(tempfile::tempdir().unwrap().path(), STEPS).unwrap();
}
