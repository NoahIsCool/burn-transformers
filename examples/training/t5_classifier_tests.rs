//! Training and native-record integration checks for `t5_classifier`.
//!
//! `train` seeds Burn Flex's process-wide random generator and draws the
//! classifier's weights from it. Tests of one binary run on parallel threads,
//! and another test drawing in between would change those weights; this
//! binary holds only the test below, so the pinned seed's draw is the one
//! trained on every run. Its margin is wide (held-out loss `0.837` to
//! `0.0003`, accuracy 100% against the 87.5% floor), but it belongs to that
//! draw: re-seeding 120 draws failed 3 (2.5%), each at the loss-halving
//! check. A test added here that draws from the generator must not run
//! concurrently with this one.

use super::*;
/// Checks disjoint held-out combinations, classifier learning, and identical logits after native-record reload.
#[test]
fn held_out_combinations_are_disjoint_and_classifier_learns_then_reloads() {
    let (train_rows, held) = dataset();
    assert_eq!((train_rows.len(), held.len()), (24, 8));
    assert!(held.iter().all(|r| !train_rows.contains(r)));
    train(tempfile::tempdir().unwrap().path(), STEPS).unwrap();
}
