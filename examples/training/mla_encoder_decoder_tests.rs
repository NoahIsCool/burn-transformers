//! Training and native-record integration checks for `mla_encoder_decoder`.
//!
//! `train` seeds Burn Flex's process-wide random generator and draws the
//! model's weights from it. Tests of one binary run on parallel threads, and
//! another test drawing in between would change those weights; this binary
//! holds only the test below, so the pinned seed's draw is the one trained on
//! every run. Its margin is wide (held-out loss `1.115` to `0.0015`, accuracy
//! 100% against the 87.5% floor), but it belongs to that draw: re-seeding 32
//! draws failed 3 (9.4%), each at the loss-halving check and two also below
//! the accuracy floor. A test added here that draws from the generator must
//! not run concurrently with this one.

use super::*;
/// Checks a disjoint complete source split, learning through MLA and the latent adapter, and native-record parity.
#[test]
fn noncausal_mla_and_hooked_latent_adapter_learn_and_reload() {
    let (train_rows, held) = dataset();
    assert_eq!(train_rows.len() + held.len(), 64);
    assert!(held.iter().all(|r| !train_rows.contains(r)));
    train(tempfile::tempdir().unwrap().path(), STEPS).unwrap();
}
