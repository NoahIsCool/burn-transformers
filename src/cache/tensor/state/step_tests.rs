//! Step write sets: coverage, atomic publication, staging, and write-ahead.
//!
//! Batched and per-row recordings of the same step must publish
//! bit-identical tensors; every malformed step must leave lengths and every
//! cache tensor unchanged (write-ahead excepted, which keeps only *logical*
//! atomicity and is tested separately).

use burn::backend::Flex;
use burn::tensor::{DType, Int, Tensor, TensorData};
use core::num::NonZeroUsize;

use super::*;
use crate::cache::{CacheConfig, KvCacheStorage, NativeKvCacheSpec, RecurrentStateSpec};

/// Deterministic host backend for exact comparisons.
type TestBackend = Flex<f32, i32>;
/// Batch rows in every fixture.
const ROWS: usize = 3;
/// Key/value heads of every KV group.
const HEADS: usize = 2;
/// Channels per head of every KV group.
const DIM: usize = 2;

/// Returns the default Flex device.
fn device() -> <TestBackend as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

/// Builds a state with a full KV group, a ring KV group, and a recurrent group.
fn state(staging: usize, in_place: bool) -> TensorDecodeState<TestBackend> {
    let config = CacheConfig::new(ROWS, 12).expect("config");
    let mut state = TensorDecodeState::with_config(config, &device());
    let mut full = NativeKvCacheSpec::new(2, HEADS, DIM, KvCacheStorage::Full, DType::F32)
        .expect("geometry")
        .with_staging_capacity(staging);
    if in_place {
        full = full.with_in_place_staging();
    }
    let ring = NativeKvCacheSpec::new(1, HEADS, DIM, KvCacheStorage::SlidingRing, DType::F32)
        .expect("geometry")
        .with_sliding_window(NonZeroUsize::new(3).expect("window"))
        .with_staging_capacity(staging);
    let recurrent = RecurrentStateSpec::new(1, 1, 2, 2, 2, 2, DType::F32)
        .expect("geometry")
        .with_staging_capacity(staging);
    state.main_mut().add_native_kv(full).expect("full group");
    state.main_mut().add_native_kv(ring).expect("ring group");
    state
        .main_mut()
        .add_recurrent(recurrent)
        .expect("recurrent group");
    state
}

/// Distinct values `[rows, HEADS, block, DIM]` tagged by `tag`.
fn kv_block(rows: usize, block: usize, tag: f32) -> Tensor<TestBackend, 4> {
    let values = (0..rows * HEADS * block * DIM)
        .map(|index| tag * 1000.0 + index as f32)
        .collect::<Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [rows, HEADS, block, DIM]),
        &device(),
    )
}

/// Distinct recurrent matrices `[rows, 1, 2, 2]` tagged by `tag`.
fn matrices(rows: usize, tag: f32) -> Tensor<TestBackend, 4> {
    let values = (0..rows * 4)
        .map(|index| tag + index as f32)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [rows, 1, 2, 2]), &device())
}

/// Distinct convolution histories `[rows, 2, 2]` tagged by `tag`.
fn convolutions(rows: usize, tag: f32) -> Tensor<TestBackend, 3> {
    let values = (0..rows * 4)
        .map(|index| -tag - index as f32)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [rows, 2, 2]), &device())
}

/// Every committed tensor of the main bank plus every row's working read, on the host.
///
/// Working reads include staged proposals, so two states with equal
/// snapshots agree on everything a model can observe.
fn snapshot(state: &TensorDecodeState<TestBackend>) -> Vec<Vec<f32>> {
    let host =
        |tensor: &Tensor<TestBackend, 4>| tensor.clone().into_data().to_vec::<f32>().expect("F32");
    let bank = state.main();
    let mut tensors = Vec::new();
    for group in 0..bank.native_kv_count() {
        let cache = bank.native_kv(group).expect("group");
        for layer in 0..cache.spec().layers() {
            tensors.push(host(cache.keys(layer).expect("keys")));
            tensors.push(host(cache.values(layer).expect("values")));
            for (row, &length) in state.lengths().iter().enumerate() {
                let read = cache
                    .read_working_sequence(layer, row, length)
                    .expect("read");
                if let Some((keys, values)) = read.materialize() {
                    tensors.push(host(&keys));
                    tensors.push(host(&values));
                }
            }
        }
    }
    for group in 0..bank.recurrent_count() {
        let cache = bank.recurrent(group).expect("group");
        for layer in 0..cache.spec().layers() {
            tensors.push(host(cache.matrix(layer).expect("matrix")));
            for row in 0..ROWS {
                let read = cache.read_working_sequence(layer, row).expect("read");
                tensors.push(host(read.matrix()));
                tensors.push(
                    read.convolution()
                        .clone()
                        .into_data()
                        .to_vec()
                        .expect("F32"),
                );
            }
        }
    }
    tensors
}

/// Staged lengths of every group, layer, and row of the main bank.
fn staged(state: &TensorDecodeState<TestBackend>) -> Vec<usize> {
    let bank = state.main();
    let mut lengths = Vec::new();
    for group in 0..bank.native_kv_count() {
        let cache = bank.native_kv(group).expect("group");
        for layer in 0..cache.spec().layers() {
            lengths.extend((0..ROWS).map(|row| cache.staged_length(layer, row).expect("row")));
        }
    }
    for group in 0..bank.recurrent_count() {
        let cache = bank.recurrent(group).expect("group");
        for layer in 0..cache.spec().layers() {
            lengths.extend((0..ROWS).map(|row| cache.staged_length(layer, row).expect("row")));
        }
    }
    lengths
}

/// Records a complete step batched: one tensor per layer for every group.
fn batched_step(
    state: &TensorDecodeState<TestBackend>,
    counts: &[usize],
    block: usize,
    tag: f32,
) -> TensorStepWrites<TestBackend> {
    let mut step = state.step(counts).expect("step");
    for (group, layers) in [(0, 2), (1, 1)] {
        for layer in 0..layers {
            let offset = tag + (10 * group + layer) as f32;
            step.push_kv_batch(
                group,
                layer,
                kv_block(ROWS, block, offset),
                kv_block(ROWS, block, offset + 100.0),
            );
        }
    }
    step.push_recurrent_batch(0, 0, matrices(ROWS, tag), convolutions(ROWS, tag));
    step
}

/// Records the same step row by row from the batched tensors.
fn per_row_step(
    state: &TensorDecodeState<TestBackend>,
    counts: &[usize],
    block: usize,
    tag: f32,
) -> TensorStepWrites<TestBackend> {
    let mut step = state.step(counts).expect("step");
    for (group, layers) in [(0, 2), (1, 1)] {
        for layer in 0..layers {
            let offset = tag + (10 * group + layer) as f32;
            let keys = kv_block(ROWS, block, offset);
            let values = kv_block(ROWS, block, offset + 100.0);
            for (row, &count) in counts.iter().enumerate().filter(|(_, count)| **count > 0) {
                let slice = [row..row + 1, 0..HEADS, 0..count, 0..DIM];
                step.push_kv_row(
                    group,
                    layer,
                    row,
                    keys.clone().slice(slice.clone()),
                    values.clone().slice(slice),
                );
            }
        }
    }
    for (row, _) in counts.iter().enumerate().filter(|(_, count)| **count > 0) {
        step.push_recurrent_row(
            0,
            0,
            row,
            matrices(ROWS, tag).slice([row..row + 1, 0..1, 0..2, 0..2]),
            convolutions(ROWS, tag).slice([row..row + 1, 0..2, 0..2]),
        );
    }
    step
}

/// Batched and per-row steps publish identical tensors and lengths.
#[test]
fn batched_and_per_row_steps_publish_identical_state() {
    let mut batched = state(0, false);
    let mut per_row = state(0, false);
    for (counts, tag) in [([3, 1, 2], 1.0), ([1, 0, 1], 2.0), ([2, 2, 0], 3.0)] {
        batched
            .publish_step(batched_step(&batched, &counts, 3, tag))
            .expect("batched step");
        per_row
            .publish_step(per_row_step(&per_row, &counts, 3, tag))
            .expect("per-row step");
        assert_eq!(batched.lengths(), per_row.lengths());
        assert_eq!(snapshot(&batched), snapshot(&per_row), "counts {counts:?}");
    }
    assert_eq!(batched.lengths(), &[6, 3, 3]);
}

/// Malformed steps are rejected before any length or tensor changes.
#[test]
fn incomplete_or_inconsistent_steps_change_nothing() {
    let mut state = state(0, false);
    state
        .publish_step(batched_step(&state, &[1, 1, 1], 1, 1.0))
        .expect("prefill");
    let before = snapshot(&state);
    let counts = [1, 0, 2];

    let mut missing_layer = state.step(&counts).expect("step");
    missing_layer.push_kv_batch(0, 0, kv_block(ROWS, 2, 5.0), kv_block(ROWS, 2, 6.0));
    let mut duplicate = batched_step(&state, &counts, 2, 7.0);
    duplicate.push_kv_row(1, 0, 0, kv_block(1, 1, 8.0), kv_block(1, 1, 8.0));
    let mut inactive = per_row_step(&state, &counts, 2, 7.0);
    inactive.push_recurrent_row(0, 0, 1, matrices(1, 1.0), convolutions(1, 1.0));
    let mut wrong_count = state.step(&counts).expect("step");
    wrong_count.push_kv_row(0, 0, 0, kv_block(1, 2, 1.0), kv_block(1, 2, 1.0));
    let mut bad_group = batched_step(&state, &counts, 2, 7.0);
    bad_group.push_kv_batch(5, 0, kv_block(ROWS, 2, 1.0), kv_block(ROWS, 2, 1.0));
    let mut bad_layer = batched_step(&state, &counts, 2, 7.0);
    bad_layer.push_kv_batch(1, 4, kv_block(ROWS, 2, 1.0), kv_block(ROWS, 2, 1.0));
    let mut bad_shape = state.step(&counts).expect("step");
    for (group, layer) in [(0, 0), (0, 1), (1, 0)] {
        bad_shape.push_kv_batch(group, layer, kv_block(ROWS, 1, 1.0), kv_block(ROWS, 1, 1.0));
    }
    bad_shape.push_recurrent_batch(0, 0, matrices(ROWS, 1.0), convolutions(ROWS, 1.0));
    let over_capacity = batched_step(&state, &[1, 0, 12], 12, 1.0);

    let cases: Vec<(&str, TensorStepWrites<TestBackend>)> = vec![
        ("missing", missing_layer),
        ("duplicate", duplicate),
        ("inactive", inactive),
        ("token count", wrong_count),
        ("group", bad_group),
        ("layer", bad_layer),
        ("block", bad_shape),
        ("capacity", over_capacity),
    ];
    for (case, step) in cases {
        let error = state.publish_step(step).expect_err(case);
        match case {
            "missing" | "duplicate" | "inactive" | "token count" => assert!(
                matches!(
                    &error,
                    TensorCacheError::StepWriteCoverage { problem, .. } if problem.contains(case)
                        || (case == "missing" && problem.contains("missing"))
                ),
                "{case}: {error}"
            ),
            "group" => assert!(matches!(error, TensorCacheError::CacheOutOfBounds { .. })),
            "layer" => assert!(matches!(error, TensorCacheError::LayerOutOfBounds { .. })),
            "block" => assert!(matches!(error, TensorCacheError::AppendOutsideBlock { .. })),
            _ => assert!(matches!(error, TensorCacheError::Metadata(_)), "{error}"),
        }
        assert_eq!(state.lengths(), &[1, 1, 1], "{case}");
        assert_eq!(snapshot(&state), before, "{case}");
    }
    assert!(
        TensorCacheError::StepWriteCoverage {
            kind: "native KV",
            group: 1,
            layer: 2,
            sequence: 0,
            problem: "missing write"
        }
        .to_string()
        .contains("group 1, layer 2")
    );
    assert!(matches!(
        state.step(&[1, 1]),
        Err(TensorCacheError::Metadata(_))
    ));
}

/// Transactional steps stage atomically and commit like per-row staging.
#[test]
fn transactional_steps_stage_atomically_and_commit() {
    for in_place in [false, true] {
        let mut batched = state(2, in_place);
        let mut per_row = state(2, in_place);
        for state in [&mut batched, &mut per_row] {
            let prefill = batched_step(state, &[2, 1, 3], 3, 1.0);
            state.publish_step(prefill).expect("prefill");
        }
        let marks = [
            batched.begin_transaction().expect("transaction"),
            per_row.begin_transaction().expect("transaction"),
        ];
        let before = (snapshot(&batched), staged(&batched));

        // Row 2 would exceed the staging capacity: nothing may be staged.
        let too_long = batched_step(&batched, &[1, 0, 3], 3, 2.0);
        assert!(matches!(
            batched.publish_step(too_long),
            Err(TensorCacheError::StagingCapacityExceeded { .. })
                | Err(TensorCacheError::StepWriteCoverage { .. })
        ));
        assert_eq!((snapshot(&batched), staged(&batched)), before);
        assert_eq!(batched.working_lengths(), &[2, 1, 3]);

        for (counts, tag) in [([1, 0, 1], 3.0), ([1, 1, 1], 4.0)] {
            batched
                .publish_step(batched_step(&batched, &counts, 1, tag))
                .expect("batched staging");
            per_row
                .publish_step(per_row_step(&per_row, &counts, 1, tag))
                .expect("per-row staging");
            assert_eq!(snapshot(&batched), snapshot(&per_row), "{in_place}");
            assert_eq!(staged(&batched), staged(&per_row));
        }
        assert_eq!(batched.lengths(), &[2, 1, 3]);
        assert_eq!(batched.working_lengths(), &[4, 2, 5]);
        let [first, second] = marks;
        batched.commit_prefix(first, &[2, 0, 1]).expect("commit");
        per_row.commit_prefix(second, &[2, 0, 1]).expect("commit");
        assert_eq!(batched.lengths(), &[4, 1, 4]);
        assert_eq!(snapshot(&batched), snapshot(&per_row), "{in_place}");
    }
}

/// Multi-token speculative steps need per-token recurrent snapshots.
#[test]
fn speculative_recurrent_steps_require_per_token_snapshots() {
    let mut state = state(3, false);
    let mark = state.begin_transaction().expect("transaction");
    let terminal = batched_step(&state, &[2, 1, 0], 2, 1.0);
    assert!(matches!(
        state.publish_step(terminal),
        Err(TensorCacheError::StepWriteCoverage { problem, .. }) if problem.contains("snapshots")
    ));
    let mut snapshots = state.step(&[2, 1, 0]).expect("step");
    for (group, layer) in [(0, 0), (0, 1), (1, 0)] {
        snapshots.push_kv_batch(group, layer, kv_block(ROWS, 2, 1.0), kv_block(ROWS, 2, 2.0));
    }
    for (row, count) in [(0, 2), (1, 1)] {
        snapshots.push_recurrent_row_snapshots(
            0,
            0,
            row,
            Tensor::ones([1, count, 1, 2, 2], &device()),
            Tensor::ones([1, count, 2, 2], &device()),
        );
    }
    state.publish_step(snapshots).expect("snapshots");
    let read = state
        .main()
        .recurrent(0)
        .expect("group")
        .read_working_sequence(0, 0)
        .expect("read");
    assert_eq!(read.staged_snapshot(), Some(1));
    state.rollback(mark).expect("rollback");
}

/// Write-ahead attends over one view and publishes like an ordinary step.
#[test]
fn write_ahead_reads_through_the_step_and_publishes() {
    let mut ahead = state(0, false);
    let mut ordinary = state(0, false);
    for state in [&mut ahead, &mut ordinary] {
        let prefill = batched_step(state, &[2, 1, 3], 3, 1.0);
        state.publish_step(prefill).expect("prefill");
    }
    let counts = [1, 0, 2];
    let mut step = ahead
        .step(&counts)
        .expect("step")
        .with_alignment(TokenAlignment::LeftPadded);
    let plan = ahead.kv_read_plan_through(&step, 0).expect("plan");
    assert!(matches!(
        ahead.read_kv_through(&step, 0, 0, &plan),
        Err(TensorCacheError::BatchPlanMismatch { .. })
    ));
    for layer in 0..2 {
        let offset = 2.0 + layer as f32;
        ahead
            .write_kv_ahead(
                &mut step,
                0,
                layer,
                kv_block(ROWS, 2, offset),
                kv_block(ROWS, 2, offset + 100.0),
            )
            .expect("write ahead");
        let read = ahead.read_kv_through(&step, 0, layer, &plan).expect("read");
        assert_eq!(read.segments().len(), 1);
        assert_eq!(read.retained_positions(), &[0..3, 0..1, 0..5]);
    }
    assert!(matches!(
        ahead.write_kv_ahead(
            &mut step,
            1,
            0,
            kv_block(ROWS, 2, 0.0),
            kv_block(ROWS, 2, 0.0)
        ),
        Err(TensorCacheError::WriteAheadUnsupported { .. })
    ));
    step.push_kv_batch(1, 0, kv_block(ROWS, 2, 12.0), kv_block(ROWS, 2, 112.0));
    step.push_recurrent_batch(0, 0, matrices(ROWS, 2.0), convolutions(ROWS, 2.0));
    ahead.publish_step(step).expect("publish");

    let mut reference = ordinary
        .step(&counts)
        .expect("step")
        .with_alignment(TokenAlignment::LeftPadded);
    for (group, layer) in [(0, 0), (0, 1), (1, 0)] {
        let offset = 2.0 + (10 * group + layer) as f32;
        reference.push_kv_batch(
            group,
            layer,
            kv_block(ROWS, 2, offset),
            kv_block(ROWS, 2, offset + 100.0),
        );
    }
    reference.push_recurrent_batch(0, 0, matrices(ROWS, 2.0), convolutions(ROWS, 2.0));
    ordinary.publish_step(reference).expect("publish");
    assert_eq!(ahead.lengths(), ordinary.lengths());
    assert_eq!(snapshot(&ahead), snapshot(&ordinary));
}

/// A failed step after a write-ahead keeps every readable position unchanged.
#[test]
fn failed_write_ahead_step_keeps_logical_state() {
    let mut state = state(0, false);
    state
        .publish_step(batched_step(&state, &[2, 2, 2], 2, 1.0))
        .expect("prefill");
    let readable = |state: &TensorDecodeState<TestBackend>| {
        (0..ROWS)
            .map(|row| {
                let read = state
                    .main()
                    .native_kv(0)
                    .expect("group")
                    .read_sequence(0, row, 2)
                    .expect("read");
                let (keys, _) = read.materialize().expect("keys");
                keys.into_data().to_vec::<f32>().expect("F32")
            })
            .collect::<Vec<_>>()
    };
    let before = readable(&state);
    let mut step = state.step(&[1, 1, 1]).expect("step");
    state
        .write_kv_ahead(
            &mut step,
            0,
            0,
            kv_block(ROWS, 1, 9.0),
            kv_block(ROWS, 1, 9.0),
        )
        .expect("write ahead");
    assert!(matches!(
        state.publish_step(step),
        Err(TensorCacheError::StepWriteCoverage { .. })
    ));
    assert_eq!(state.lengths(), &[2, 2, 2]);
    assert_eq!(readable(&state), before);
    // The abandoned slots are simply overwritten by the next successful step.
    state
        .publish_step(batched_step(&state, &[1, 1, 1], 1, 4.0))
        .expect("next step");
    assert_eq!(state.lengths(), &[3, 3, 3]);
}

/// Inside a transaction only in-place groups write ahead, and their staging is deferred.
#[test]
fn transactional_write_ahead_requires_in_place_staging() {
    let mut separate = state(2, false);
    separate.begin_transaction().expect("transaction");
    let mut step = separate.step(&[1, 1, 1]).expect("step");
    assert!(matches!(
        separate.write_kv_ahead(&mut step, 0, 0, kv_block(ROWS, 1, 1.0), kv_block(ROWS, 1, 1.0)),
        Err(TensorCacheError::WriteAheadUnsupported { reason, .. }) if reason.contains("in-place")
    ));

    let mut in_place = state(2, true);
    let mark = in_place.begin_transaction().expect("transaction");
    let mut step = in_place.step(&[1, 0, 1]).expect("step");
    for layer in 0..2 {
        in_place
            .write_kv_ahead(
                &mut step,
                0,
                layer,
                kv_block(ROWS, 1, 1.0),
                kv_block(ROWS, 1, 1.0),
            )
            .expect("write ahead");
    }
    // Staged lengths stay unpublished until the step is.
    let full = in_place.main().native_kv(0).expect("group");
    assert_eq!(
        (0..ROWS)
            .map(|row| full.staged_length(0, row).expect("row"))
            .collect::<Vec<_>>(),
        [0, 0, 0]
    );
    step.push_kv_batch(1, 0, kv_block(ROWS, 1, 2.0), kv_block(ROWS, 1, 2.0));
    step.push_recurrent_batch(0, 0, matrices(ROWS, 1.0), convolutions(ROWS, 1.0));
    in_place.publish_step(step).expect("publish");
    let full = in_place.main().native_kv(0).expect("group");
    assert_eq!(
        (0..ROWS)
            .map(|row| full.staged_length(1, row).expect("row"))
            .collect::<Vec<_>>(),
        [1, 0, 1]
    );
    assert_eq!(in_place.working_lengths(), &[1, 0, 1]);
    in_place.commit_prefix(mark, &[1, 0, 0]).expect("commit");
    assert_eq!(in_place.lengths(), &[1, 0, 0]);
}

/// A step's token segment joins a history read; the result equals the per-row concatenation.
#[test]
fn step_token_segments_extend_history_reads() {
    let mut state = state(0, false);
    state
        .publish_step(batched_step(&state, &[2, 1, 0], 2, 1.0))
        .expect("prefill");
    let step = state
        .step(&[1, 2, 1])
        .expect("step")
        .with_alignment(TokenAlignment::LeftPadded);
    let plan = state.kv_read_plan(StepBank::Main, 0).expect("plan");
    let history = state
        .main()
        .native_kv(0)
        .expect("group")
        .read_batch(0, &plan)
        .expect("history");
    let new_keys = kv_block(ROWS, 2, 5.0);
    let segment = state
        .step_token_segment(&step, new_keys.clone(), kv_block(ROWS, 2, 6.0))
        .expect("segment");
    let positions: Vec<i32> = segment
        .key_positions()
        .clone()
        .into_data()
        .to_vec()
        .expect("I32");
    let valid: Vec<bool> = segment
        .key_valid()
        .expect("left padding leaves gaps")
        .clone()
        .into_data()
        .to_vec()
        .expect("Bool");
    assert_eq!(valid, [false, true, true, true, false, true]);
    assert_eq!(
        [positions[1], positions[2], positions[3], positions[5]],
        [2, 1, 2, 0]
    );
    let joined = history.with_segment(segment).expect("join");
    let all = joined.materialize().expect("keys");
    assert_eq!(all.keys().dims(), [ROWS, HEADS, 4, DIM]);
    let query =
        Tensor::<TestBackend, 2, Int>::from_ints(TensorData::new(vec![2, 2, 0], [3, 1]), &device());
    let mask = all.exclusion_mask(&query, 1, None).expect("mask");
    let mask: Vec<bool> = mask.into_data().to_vec().expect("Bool");
    // Row 0 sees history 0..2 and its token at 2; row 2 sees only its token at 0.
    assert_eq!(&mask[0..4], [false, false, true, false]);
    assert_eq!(&mask[8..12], [true, true, true, false]);
    let other = Tensor::<TestBackend, 4>::zeros([2, HEADS, 1, DIM], &device());
    let wrong = state.step_token_segment(&step, other.clone(), other);
    assert!(matches!(wrong, Err(TensorCacheError::Metadata(_))));
}

/// The MTP stack publishes its own steps against its own cursor.
#[test]
fn mtp_steps_advance_only_the_auxiliary_cursor() {
    let config = CacheConfig::new(ROWS, 8).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::with_config(config, &device());
    assert!(matches!(
        state.mtp_step(&[1, 1, 1]),
        Err(TensorCacheError::MtpNotEnabled)
    ));
    let spec =
        NativeKvCacheSpec::new(1, HEADS, DIM, KvCacheStorage::Full, DType::F32).expect("geometry");
    state
        .enable_mtp()
        .expect("MTP")
        .add_native_kv(spec)
        .expect("group");
    let mut step = state.mtp_step(&[2, 0, 1]).expect("step");
    assert_eq!(step.bank(), StepBank::Mtp);
    assert_eq!(step.token_counts(), &[2, 0, 1]);
    step.push_kv_batch(0, 0, kv_block(ROWS, 2, 1.0), kv_block(ROWS, 2, 2.0));
    state.publish_step(step).expect("publish");
    assert_eq!(state.lengths(), &[0, 0, 0]);
    assert_eq!(state.mtp_lengths(), Some([2, 0, 1].as_slice()));
    let plan = state.kv_read_plan(StepBank::Mtp, 0).expect("plan");
    assert_eq!(plan.retained_positions(), &[0..2, 0..0, 0..1]);
    assert!(
        TensorCacheError::WriteAheadUnsupported {
            storage: "sliding ring",
            reason: "x"
        }
        .to_string()
        .contains("write ahead")
    );
}

/// Staging a layer outside the step API makes the next step's cursors disagree.
#[test]
fn steps_reject_layers_staged_outside_the_step() {
    for stray_kv in [true, false] {
        let mut state = state(3, false);
        let mark = state.begin_transaction().expect("transaction");
        if stray_kv {
            state
                .main_mut()
                .native_kv_mut(1)
                .expect("ring")
                .stage_sequence(0, 0, kv_block(1, 1, 1.0), kv_block(1, 1, 1.0))
                .expect("stray staging");
        } else {
            state
                .main_mut()
                .recurrent_mut(0)
                .expect("recurrent")
                .stage_sequence_snapshots(
                    0,
                    0,
                    Tensor::ones([1, 1, 1, 2, 2], &device()),
                    Tensor::ones([1, 1, 2, 2], &device()),
                )
                .expect("stray staging");
        }
        let before = (snapshot(&state), staged(&state));
        for batched in [true, false] {
            let step = if batched {
                batched_step(&state, &[1, 1, 1], 1, 2.0)
            } else {
                per_row_step(&state, &[1, 1, 1], 1, 2.0)
            };
            assert!(matches!(
                state.publish_step(step),
                Err(TensorCacheError::TransactionCursorMismatch { sequence: 0, .. })
            ));
            assert_eq!((snapshot(&state), staged(&state)), before);
        }
        state.rollback(mark).expect("rollback");
    }
}

/// Latent groups join steps, write-ahead, and token segments like KV groups.
#[test]
fn latent_groups_publish_through_steps() {
    let config = CacheConfig::new(2, 8).expect("config");
    let latent_spec = NativeKvCacheSpec::latent(2, 4, KvCacheStorage::Full, DType::F32)
        .expect("geometry")
        .with_staging_capacity(2)
        .with_in_place_staging();
    let latent = |rows: usize, tokens: usize, tag: f32| {
        let data = (0..rows * tokens * 4)
            .map(|index| tag + index as f32)
            .collect::<Vec<_>>();
        Tensor::<TestBackend, 3>::from_data(TensorData::new(data, [rows, tokens, 4]), &device())
    };
    let mut batched = TensorDecodeState::<TestBackend>::with_config(config, &device());
    let mut per_row = TensorDecodeState::<TestBackend>::with_config(config, &device());
    for state in [&mut batched, &mut per_row] {
        state
            .main_mut()
            .add_native_kv(latent_spec)
            .expect("latent group");
    }

    let mut step = batched.step(&[2, 1]).expect("step");
    let plan = batched.kv_read_plan_through(&step, 0).expect("plan");
    for layer in 0..2 {
        batched
            .write_latent_ahead(&mut step, 0, layer, latent(2, 2, layer as f32))
            .expect("write ahead");
        let read = batched
            .read_kv_through(&step, 0, layer, &plan)
            .expect("read");
        assert_eq!(read.materialize().expect("keys").latent().dims(), [2, 2, 4]);
    }
    // Layer 0 is already covered: a second write is refused before any tensor work.
    assert!(matches!(
        batched.write_latent_ahead(&mut step, 0, 0, latent(2, 2, 5.0)),
        Err(TensorCacheError::StepWriteCoverage {
            problem: "duplicate write",
            ..
        })
    ));
    // A rejected write-ahead of another step leaves `step` current.
    let mut other = batched.step(&[2, 1]).expect("step");
    assert!(matches!(
        batched.write_kv_ahead(&mut other, 0, 0, kv_block(2, 2, 0.0), kv_block(2, 2, 0.0)),
        Err(TensorCacheError::PlaneKindMismatch { .. })
    ));
    batched.publish_step(step).expect("publish");

    let mut reference = per_row.step(&[2, 1]).expect("step");
    for layer in 0..2 {
        let block = latent(2, 2, layer as f32);
        reference.push_latent_row(0, layer, 0, block.clone().slice([0..1, 0..2, 0..4]));
        reference.push_latent_row(0, layer, 1, block.slice([1..2, 0..1, 0..4]));
    }
    per_row.publish_step(reference).expect("publish");
    assert_eq!(snapshot(&batched), snapshot(&per_row));

    // A speculative latent step stages in place and joins a token segment.
    let mark = batched.begin_transaction().expect("transaction");
    let mut proposal = batched.step(&[1, 1]).expect("step");
    let segment = batched
        .step_latent_segment(&proposal, latent(2, 1, 9.0))
        .expect("segment");
    assert!(segment.key_valid().is_none());
    for layer in 0..2 {
        proposal.push_latent_batch(0, layer, latent(2, 1, 9.0));
    }
    let mut wrong = batched.step(&[1, 1]).expect("step");
    for layer in 0..2 {
        wrong.push_kv_batch(0, layer, kv_block(2, 1, 0.0), kv_block(2, 1, 0.0));
    }
    assert!(matches!(
        batched.publish_step(wrong),
        Err(TensorCacheError::PlaneKindMismatch { .. })
    ));
    batched.publish_step(proposal).expect("stage");
    assert_eq!(batched.working_lengths(), &[3, 2]);
    batched.commit_prefix(mark, &[1, 0]).expect("commit");
    assert_eq!(batched.lengths(), &[3, 1]);
}

/// State-level reads use the state's cursors and reject layers out of step.
#[test]
fn checked_reads_use_state_cursors_and_detect_stray_staging() {
    let mut state = state(3, false);
    state
        .publish_step(batched_step(&state, &[2, 1, 1], 2, 1.0))
        .expect("prefill");
    let mark = state.begin_transaction().expect("transaction");
    state
        .publish_step(batched_step(&state, &[1, 1, 1], 1, 2.0))
        .expect("proposal");
    let read = state
        .read_kv(StepBank::Main, 0, 1, 0)
        .expect("checked read");
    assert_eq!(read.retained_positions(), 0..3);
    let recurrent = state
        .read_recurrent(StepBank::Main, 0, 0, 2)
        .expect("checked recurrent read");
    assert_eq!(recurrent.staged_snapshot(), Some(0));
    // Stage one more token for one layer only, outside any step.
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("group")
        .stage_sequence(0, 0, kv_block(1, 1, 3.0), kv_block(1, 1, 3.0))
        .expect("stray staging");
    state
        .main_mut()
        .recurrent_mut(0)
        .expect("group")
        .stage_sequence_snapshots(
            0,
            1,
            Tensor::ones([1, 1, 1, 2, 2], &device()),
            Tensor::ones([1, 1, 2, 2], &device()),
        )
        .expect("stray snapshot");
    assert!(matches!(
        state.read_kv(StepBank::Main, 0, 0, 0),
        Err(TensorCacheError::TransactionCursorMismatch {
            expected: 1,
            actual: 2,
            ..
        })
    ));
    assert!(state.read_kv(StepBank::Main, 0, 1, 0).is_ok());
    assert!(matches!(
        state.read_recurrent(StepBank::Main, 0, 0, 1),
        Err(TensorCacheError::TransactionCursorMismatch {
            expected: 1,
            actual: 2,
            ..
        })
    ));
    assert!(matches!(
        state.read_kv(StepBank::Main, 0, 0, ROWS),
        Err(TensorCacheError::Metadata(_))
    ));
    assert!(matches!(
        state.read_kv(StepBank::Mtp, 0, 0, 0),
        Err(TensorCacheError::MtpNotEnabled)
    ));
    state.rollback(mark).expect("rollback");
    assert!(state.read_kv(StepBank::Main, 0, 0, 0).is_ok());
}

/// Writes layers 0 and 1 of the full group ahead and pushes the ring and recurrent groups.
///
/// The result is a complete step for the [`state`] fixture.
fn written_ahead_step(
    state: &mut TensorDecodeState<TestBackend>,
    counts: &[usize],
    tag: f32,
) -> TensorStepWrites<TestBackend> {
    let block = counts.iter().copied().max().unwrap_or(1).max(1);
    let mut step = state.step(counts).expect("step");
    for layer in 0..2 {
        let offset = tag + layer as f32;
        state
            .write_kv_ahead(
                &mut step,
                0,
                layer,
                kv_block(ROWS, block, offset),
                kv_block(ROWS, block, offset + 100.0),
            )
            .expect("write ahead");
    }
    step.push_kv_batch(
        1,
        0,
        kv_block(ROWS, block, tag + 10.0),
        kv_block(ROWS, block, tag + 110.0),
    );
    step.push_recurrent_batch(0, 0, matrices(ROWS, tag), convolutions(ROWS, tag));
    step
}

/// Asserts that an operation on a step reports it stale.
fn assert_stale<T: core::fmt::Debug>(result: TensorCacheResult<T>, reason: &str) {
    match result {
        Err(TensorCacheError::StaleStep { reason: actual, .. }) => {
            assert!(actual.contains(reason), "{actual}");
        }
        other => panic!("expected a stale step ({reason}), got {other:?}"),
    }
}

/// A write-ahead made outside a transaction cannot be published into one.
///
/// Its tokens went to committed slots, but a separate-staging group reads
/// proposals from its staging buffer, which the write never touched:
/// publishing would record proposals whose data is missing and a later
/// commit would copy zeros over the real tokens.
#[test]
fn write_ahead_cannot_be_published_into_a_later_transaction() {
    let mut state = state(2, false);
    state
        .publish_step(batched_step(&state, &[1, 1, 1], 1, 1.0))
        .expect("prefill");
    let step = written_ahead_step(&mut state, &[1, 1, 1], 7.0);
    let mark = state.begin_transaction().expect("transaction");
    let before = (snapshot(&state), staged(&state));
    let plan = state.kv_read_plan(StepBank::Main, 0).expect("history plan");
    assert_stale(state.read_kv_through(&step, 0, 0, &plan), "bank changed");
    assert_stale(state.kv_read_plan_through(&step, 0), "bank changed");
    assert_stale(
        state.step_token_segment(&step, kv_block(ROWS, 1, 0.0), kv_block(ROWS, 1, 0.0)),
        "bank changed",
    );
    assert_stale(state.publish_step(step), "bank changed");
    assert_eq!(state.working_lengths(), &[1, 1, 1]);
    assert_eq!((snapshot(&state), staged(&state)), before);
    state.rollback(mark).expect("rollback");
}

/// A write-ahead does not survive a reset that zeroed its slots.
#[test]
fn write_ahead_cannot_be_published_after_reset() {
    let mut state = state(0, false);
    let step = written_ahead_step(&mut state, &[1, 1, 1], 9.0);
    state.reset().expect("reset");
    let before = snapshot(&state);
    assert_stale(state.publish_step(step), "bank changed");
    assert_eq!(state.lengths(), &[0, 0, 0]);
    assert_eq!(snapshot(&state), before);
}

/// A step, with or without write-ahead, only publishes into the state that created it.
#[test]
fn steps_cannot_be_published_into_another_state() {
    let mut origin = state(0, false);
    let mut target = state(0, false);
    let before = snapshot(&target);
    let ahead = written_ahead_step(&mut origin, &[1, 1, 1], 9.0);
    assert_stale(target.publish_step(ahead), "another decode state");
    let pushed = batched_step(&origin, &[1, 1, 1], 1, 2.0);
    assert_stale(target.publish_step(pushed), "another decode state");
    assert_eq!(target.lengths(), &[0, 0, 0]);
    assert_eq!(snapshot(&target), before);
    assert!(
        TensorCacheError::StaleStep {
            bank: "main",
            reason: "x"
        }
        .to_string()
        .contains("main step is stale")
    );
}

/// Every operation that can move a bank's cursors or tensors invalidates its outstanding steps.
#[test]
fn lifecycle_operations_invalidate_outstanding_steps() {
    type Operation = fn(&mut TensorDecodeState<TestBackend>);
    let operations: [(&str, Operation); 8] = [
        ("advance", |state| {
            state.advance(&[1, 0, 0]).expect("advance")
        }),
        ("begin", |state| {
            state.begin_transaction().expect("transaction");
        }),
        ("reorder", |state| {
            state.reorder(&[2, 1, 0]).expect("reorder")
        }),
        ("truncate", |state| {
            state.truncate(&[0, 2, 2]).expect("truncate")
        }),
        ("reset", |state| state.reset().expect("reset")),
        ("mutable access", |state| {
            let _ = state.main_mut();
        }),
        ("another publish", |state| {
            let other = batched_step(state, &[1, 1, 1], 1, 5.0);
            state.publish_step(other).expect("publish");
        }),
        ("another write-ahead", |state| {
            let _abandoned = written_ahead_step(state, &[1, 1, 1], 6.0);
        }),
    ];
    for (name, operation) in operations {
        let mut state = state(2, false);
        state
            .publish_step(batched_step(&state, &[2, 2, 2], 2, 1.0))
            .expect("prefill");
        let step = batched_step(&state, &[1, 1, 1], 1, 3.0);
        operation(&mut state);
        assert!(
            matches!(
                state.publish_step(step),
                Err(TensorCacheError::StaleStep { .. })
            ),
            "{name}"
        );
    }

    // Commit and rollback end the transaction a step was staged against.
    for commit in [true, false] {
        let mut state = state(2, false);
        let mark = state.begin_transaction().expect("transaction");
        let step = batched_step(&state, &[1, 1, 1], 1, 3.0);
        if commit {
            state.commit_prefix(mark, &[0, 0, 0]).expect("commit");
        } else {
            state.rollback(mark).expect("rollback");
        }
        assert_stale(state.publish_step(step), "bank changed");
    }

    // A rejected operation changes nothing and leaves steps current.
    let mut state = state(2, false);
    let step = batched_step(&state, &[1, 1, 1], 1, 3.0);
    assert!(state.truncate(&[1, 0, 0]).is_err());
    assert!(state.reorder(&[5, 0, 0]).is_err());
    state.publish_step(step).expect("still current");
}

/// Of two steps on one bank, only the one that wrote ahead last stays valid.
#[test]
fn a_write_ahead_claims_the_bank_from_other_steps() {
    let mut state = state(0, false);
    let mut first = state.step(&[1, 1, 1]).expect("step");
    let mut second = state.step(&[1, 1, 1]).expect("step");
    state
        .write_kv_ahead(
            &mut first,
            0,
            0,
            kv_block(ROWS, 1, 1.0),
            kv_block(ROWS, 1, 2.0),
        )
        .expect("first write-ahead");
    assert_stale(
        state.write_kv_ahead(
            &mut second,
            0,
            0,
            kv_block(ROWS, 1, 3.0),
            kv_block(ROWS, 1, 4.0),
        ),
        "bank changed",
    );
    assert_stale(state.publish_step(second), "bank changed");
    state
        .write_kv_ahead(
            &mut first,
            0,
            1,
            kv_block(ROWS, 1, 5.0),
            kv_block(ROWS, 1, 6.0),
        )
        .expect("the claiming step keeps writing");
    first.push_kv_batch(1, 0, kv_block(ROWS, 1, 7.0), kv_block(ROWS, 1, 8.0));
    first.push_recurrent_batch(0, 0, matrices(ROWS, 1.0), convolutions(ROWS, 1.0));
    state.publish_step(first).expect("publish");
    assert_eq!(state.lengths(), &[1, 1, 1]);
    let keys = state
        .read_kv(StepBank::Main, 0, 0, 0)
        .expect("read")
        .materialize()
        .expect("one key")
        .0;
    assert_eq!(
        keys.into_data().to_vec::<f32>().expect("F32"),
        kv_block(1, 1, 1.0)
            .into_data()
            .to_vec::<f32>()
            .expect("F32"),
        "the first step's write-ahead was not overwritten"
    );
}

/// A layer can be written ahead once per step and never on top of a pushed update.
#[test]
fn duplicate_write_ahead_is_refused_before_writing() {
    let mut state = state(0, false);
    let mut step = state.step(&[1, 0, 1]).expect("step");
    step.push_kv_batch(0, 0, kv_block(ROWS, 1, 1.0), kv_block(ROWS, 1, 1.0));
    state
        .write_kv_ahead(
            &mut step,
            0,
            1,
            kv_block(ROWS, 1, 2.0),
            kv_block(ROWS, 1, 2.0),
        )
        .expect("first write-ahead");
    let before = snapshot(&state);
    for layer in [0, 1] {
        assert!(matches!(
            state.write_kv_ahead(
                &mut step,
                0,
                layer,
                kv_block(ROWS, 1, 3.0),
                kv_block(ROWS, 1, 3.0),
            ),
            Err(TensorCacheError::StepWriteCoverage {
                sequence: 0,
                problem: "duplicate write",
                ..
            })
        ));
        assert_eq!(snapshot(&state), before, "layer {layer}");
    }
    step.push_kv_batch(1, 0, kv_block(ROWS, 1, 4.0), kv_block(ROWS, 1, 4.0));
    step.push_recurrent_batch(0, 0, matrices(ROWS, 1.0), convolutions(ROWS, 1.0));
    state.publish_step(step).expect("publish");
    assert_eq!(state.lengths(), &[1, 0, 1]);

    // A push recorded after a write-ahead of the same layer fails publication.
    let mut step = written_ahead_step(&mut state, &[1, 1, 1], 5.0);
    step.push_kv_batch(0, 1, kv_block(ROWS, 1, 6.0), kv_block(ROWS, 1, 6.0));
    assert!(matches!(
        state.publish_step(step),
        Err(TensorCacheError::StepWriteCoverage {
            problem: "duplicate write",
            ..
        })
    ));
    assert_eq!(state.lengths(), &[1, 0, 1]);
}

/// Steps of one bank survive operations on the other bank.
#[test]
fn bank_versions_are_independent() {
    let config = CacheConfig::new(ROWS, 8).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::with_config(config, &device());
    let spec =
        NativeKvCacheSpec::new(1, HEADS, DIM, KvCacheStorage::Full, DType::F32).expect("geometry");
    state.main_mut().add_native_kv(spec).expect("main group");
    state
        .enable_mtp()
        .expect("MTP")
        .add_native_kv(spec)
        .expect("MTP group");
    let mut main = state.step(&[1, 1, 1]).expect("main step");
    main.push_kv_batch(0, 0, kv_block(ROWS, 1, 1.0), kv_block(ROWS, 1, 1.0));
    let mut mtp = state.mtp_step(&[1, 0, 1]).expect("MTP step");
    mtp.push_kv_batch(0, 0, kv_block(ROWS, 1, 2.0), kv_block(ROWS, 1, 2.0));
    state.advance_mtp(&[0, 0, 0]).expect("MTP advance");
    state
        .publish_step(main)
        .expect("main step survives MTP operations");
    assert_stale(state.publish_step(mtp), "bank changed");
    let mut mtp = state.mtp_step(&[1, 0, 1]).expect("MTP step");
    mtp.push_kv_batch(0, 0, kv_block(ROWS, 1, 2.0), kv_block(ROWS, 1, 2.0));
    state.advance(&[0, 0, 0]).expect("main advance");
    state
        .publish_step(mtp)
        .expect("MTP step survives main operations");
    assert_eq!(state.lengths(), &[1, 1, 1]);
    assert_eq!(state.mtp_lengths(), Some([1, 0, 1].as_slice()));
}
