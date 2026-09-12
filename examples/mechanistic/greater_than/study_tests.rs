//! Checks greater-than probability accounting, cutoff metrics, recovery, and fixture alignment.
use super::*;
/// Checks numeric subsets keep full-vocabulary mass, remain shift invariant, and reject NaN logits.
#[test]
fn numeric_subset_keeps_full_vocabulary_probability_mass() {
    let p = numeric_probabilities(&[0.0, 0.0, 0.0, 0.0], &[1, 3]).unwrap();
    assert_eq!(p, [0.25, 0.25]);
    let shifted = numeric_probabilities(&[1000.0; 4], &[1, 3]).unwrap();
    assert_eq!(p, shifted);
    assert!(numeric_probabilities(&[f32::NAN], &[0]).is_err());
}
/// Checks the clean suffix controls signed mass and neighbor sharpness, and rejects invalid cutoffs.
#[test]
fn clean_suffix_and_neighbor_sharpness_use_original_cutoff() {
    let mut p = vec![0.0; 100];
    p[31] = 0.02;
    p[32] = 0.03;
    p[33] = 0.08;
    let (pd, sharpness, mass) = metrics(&p, 32).unwrap();
    assert!((pd - 0.03).abs() < 1e-12);
    assert!((sharpness - 0.06).abs() < 1e-12);
    assert!((mass - 0.13).abs() < 1e-12);
    assert!(metrics(&p, 1).is_err());
}
/// Checks signed recovery for negative gaps, undefined zero gaps, and sample-standard-deviation accounting.
#[test]
fn recovery_keeps_negative_gaps_and_declares_negligible_ones_undefined() {
    assert_eq!(recovery(0.6, 0.8, 0.4, 1e-6), Some(0.4999999999999999));
    assert_eq!(recovery(0.4, 0.2, 0.6, 1e-6), Some(0.49999999999999994));
    assert_eq!(recovery(0.4, 0.4, 0.4, 1e-6), None);
    assert_eq!(statistics(&[1.0, 2.0, 3.0])["sd"], 1.0);
}
/// Checks the pinned clean/corrupt fixture satisfies alignment and disjoint clean-prompt split constraints.
#[test]
fn frozen_fixture_preserves_alignment_and_clean_split_disjointness() {
    let fixture: Fixture = serde_json::from_slice(include_bytes!("fixture.json")).unwrap();
    validate_fixture(&fixture).unwrap();
}
