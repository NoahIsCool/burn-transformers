//! Training and native-record integration checks for `custom_composition`.

use super::*;
#[test]
fn custom_mixer_and_width_adapters_learn_and_survive_native_records() {
    let (train_rows, held) = dataset();
    assert_eq!(train_rows.len() + held.len(), 64);
    assert!(held.iter().all(|r| !train_rows.contains(r)));
    train(tempfile::tempdir().unwrap().path(), STEPS).unwrap();
}
