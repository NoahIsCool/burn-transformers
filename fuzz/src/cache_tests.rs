//! Deterministic regression seeds for the cache fuzz harness.

use super::*;

/// Appends, failed mutation under a mark, rejected commit, rollback and stale marks interact.
#[test]
fn transactional_sequence_restores_reference_state() {
    exercise(&[
        0, 7, 0, 0, 3, 0, 6, 0, 0, 0, 0, 0, 2, 0, 7, 0, 3, 0, 8, 0, 0, 0, 9, 0, 0, 0,
    ]);
    exercise(&[255; 128]);
    for start in 0..12 {
        for count in 0..12 {
            check_ring(start, count, 4);
        }
    }
    check_ring(usize::MAX, 1, 4);
}

/// Two-layer, two-head write-ahead steps through every ending (see `fuzz/seeds`).
///
/// Header: rows 3, capacity 12, windows 2 and 5, staging 3, recurrent, two
/// layers, width 2. Each operation is `[operation, count x3, selector]`:
/// write-ahead (7) published (selector % 4 < 2), abandoned (2), or
/// interfered with (3: begin, rollback, or an empty step), mixed with
/// ordinary steps, begin, and commit.
const WRITE_AHEAD_SCRIPT: &[u8] = include_bytes!("../seeds/cache_transactions/tensor_write_ahead");

/// Write-ahead endings match the oracle and stale steps are refused.
#[test]
fn tensor_write_ahead_sequences_match_the_history_oracle() {
    exercise_tensor_state(WRITE_AHEAD_SCRIPT);
    // The same script with one layer and width 1 takes the scalar paths.
    let mut scalar = WRITE_AHEAD_SCRIPT.to_vec();
    scalar[6] = 0;
    scalar[7] = 0;
    exercise_tensor_state(&scalar);
}

/// Steps, transactions, rewinds, and resets over every group kind agree with the oracle.
#[test]
fn tensor_state_sequences_match_the_history_oracle() {
    exercise_tensor_state(&[]);
    exercise_tensor_state(&[255; 384]);
    exercise_tensor_state(&(0..=255).collect::<Vec<_>>());
    // rows 3, capacity 12, windows 2 and 5, staging 3, recurrent, one layer,
    // width 1; then step, begin, step, commit, step, rollback, reorder,
    // truncate, reset.
    let mut script = vec![2, 10, 1, 4, 2, 0, 0, 0];
    for operation in [0, 1, 0, 2, 0, 3, 4, 5, 6, 0, 1, 0, 0, 2] {
        script.extend([operation, 2, 1, 3, operation % 2]);
    }
    exercise_tensor_state(&script);
    exercise_tensor_state(WRITE_AHEAD_SCRIPT);
    for seed in 0..64_u8 {
        let input: Vec<u8> = (0..=255_u8)
            .chain(0..44)
            .map(|index| seed.wrapping_mul(31).wrapping_add(index.wrapping_mul(7)))
            .collect();
        exercise_tensor_state(&input);
    }
    // A linear congruential generator covers many more operation mixes.
    let mut seed = 0x2545_f491_u32;
    for _ in 0..256 {
        let input: Vec<u8> = (0..384)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                seed.to_le_bytes()[3]
            })
            .collect();
        exercise_tensor_state(&input);
    }
}

/// The MTP bank follows its own steps and cursor through every lifecycle operation.
#[test]
fn mtp_bank_follows_steps_commits_reorder_truncate_and_reset() {
    let geometry = Geometry {
        rows: 2,
        capacity: 10,
        windows: [1, 4],
        staging: 2,
        recurrent: false,
        mtp: true,
        layers: 1,
        width: 1,
    };
    let mut harness = TensorHarness::new(geometry);
    assert!(harness.publish_mtp(&[2, 1], true));
    harness.check();
    assert!(harness.publish(&[1, 1], false));
    assert!(harness.begin());
    assert!(harness.publish_mtp(&[1, 2], false));
    assert!(harness.publish(&[2, 1], true));
    harness.check();
    // Staged MTP tokens need their own accepted prefix, which must fit the
    // MTP proposals `[1, 2]` independently of the main proposals `[2, 1]`.
    assert!(!harness.commit(&[1, 1], None));
    assert!(!harness.commit(&[1, 1], Some(&[2, 0])));
    assert!(!harness.commit(&[2, 2], Some(&[0, 2])));
    harness.check();
    assert!(harness.commit(&[1, 1], Some(&[0, 2])));
    harness.check();
    assert!(harness.reorder(&[1, 0]));
    harness.check();
    assert!(harness.truncate_mtp(&[1, 0]));
    harness.check();
    assert!(harness.begin());
    // Three MTP tokens exceed the staging capacity of two.
    assert!(!harness.publish_mtp(&[3, 0], true));
    assert!(!harness.truncate_mtp(&[0, 0]));
    assert!(harness.rollback());
    harness.check();
    assert!(harness.reset());
    harness.check();
    let mut plain = TensorHarness::new(Geometry {
        mtp: false,
        ..geometry
    });
    assert!(!plain.publish_mtp(&[1, 1], true));
    assert!(!plain.truncate_mtp(&[0, 0]));
    // An MTP prefix without an MTP bank is refused; the plain commit succeeds.
    assert!(plain.begin());
    assert!(!plain.commit(&[0, 0], Some(&[0, 0])));
    assert!(plain.commit(&[0, 0], None));
    plain.check();
    exercise_tensor_state(include_bytes!("../seeds/cache_transactions/tensor_mtp"));
}

/// Geometry of the single-row MTP scripts: capacity 10, staging 3, no recurrent group.
const SINGLE_ROW_MTP: Geometry = Geometry {
    rows: 1,
    capacity: 10,
    windows: [2, 4],
    staging: 3,
    recurrent: false,
    mtp: true,
    layers: 1,
    width: 1,
};

/// A single row commits different main and MTP prefixes and each bank keeps its own.
///
/// With one row, reversing the main prefix (the harness's earlier MTP prefix)
/// reproduces it, so a library that applied the main prefix to the MTP bank
/// went undetected; the independent prefix tells the two apart.
#[test]
fn single_row_commits_independent_main_and_mtp_prefixes() {
    let mut harness = TensorHarness::new(SINGLE_ROW_MTP);
    assert!(harness.publish(&[2], true));
    assert!(harness.publish_mtp(&[1], false));
    assert!(harness.begin());
    assert!(harness.publish(&[2], false));
    assert!(harness.publish_mtp(&[3], true));
    harness.check();
    assert!(harness.commit(&[2], Some(&[1])));
    harness.check();
    assert_eq!(harness.state.lengths(), [4]);
    assert_eq!(harness.state.mtp_lengths(), Some(&[2][..]));
    assert!(harness.begin());
    assert!(harness.publish(&[1], true));
    assert!(harness.publish_mtp(&[2], false));
    assert!(harness.commit(&[0], Some(&[2])));
    harness.check();
    assert_eq!(harness.state.lengths(), [4]);
    assert_eq!(harness.state.mtp_lengths(), Some(&[4][..]));
}

/// The MTP prefix comes from selector bits 1..=6, clamped to the staged length by bit 7.
#[test]
fn mtp_prefixes_decode_from_their_own_selector_bits() {
    let mut harness = TensorHarness::new(Geometry {
        rows: 3,
        ..SINGLE_ROW_MTP
    });
    assert_eq!(harness.mtp_prefix(0b0100_0110), [3, 0, 2]);
    // Outside a transaction nothing is staged, so clamping gives zeros.
    assert_eq!(harness.mtp_prefix(0b1100_0110), [0, 0, 0]);
    assert!(harness.begin());
    assert!(harness.publish_mtp(&[1, 3, 0], true));
    let clamped = harness.mtp_prefix(0b1111_1110);
    assert_eq!(clamped, [1, 3, 0]);
    assert!(!harness.commit(&[0, 0, 0], Some(&harness.mtp_prefix(0b0111_1110))));
    assert!(harness.commit(&[0, 0, 0], Some(&clamped)));
    harness.check();
}

/// The per-row MTP read is compared by value, not only by its positions.
///
/// Replacing the oracle's MTP tokens with different values of the same length
/// keeps every position range intact, so only a value comparison fails.
#[test]
#[should_panic(expected = "MTP row 0 group 0 layer 0 keys")]
fn mtp_reads_are_compared_by_value() {
    let mut harness = TensorHarness::new(SINGLE_ROW_MTP);
    assert!(harness.publish_mtp(&[2], true));
    harness.check();
    let mtp = harness.mtp.as_mut().expect("MTP bank");
    for value in &mut mtp.committed[0] {
        *value += 1000.0;
    }
    // The batched read would also catch it; check the per-row read alone.
    let read = harness
        .state
        .read_kv(StepBank::Mtp, 0, 0, 0)
        .expect("checked MTP read");
    let tokens = harness.mtp.as_ref().expect("MTP bank").working(0);
    check_row_read(&read, &tokens, 0, (1, 1), 0, 0, "MTP row 0");
}
