//! Training and native-record integration checks for `mla_encoder_decoder`.

use super::*;
/// Checks a disjoint complete source split, learning through MLA and the latent adapter, and native-record parity.
#[test]
fn noncausal_mla_and_hooked_latent_adapter_learn_and_reload() {
    let (train_rows, held) = dataset();
    assert_eq!(train_rows.len() + held.len(), 64);
    assert!(held.iter().all(|r| !train_rows.contains(r)));
    train(tempfile::tempdir().unwrap().path(), STEPS).unwrap();
}
