use super::*;

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("test values are non-zero")
}

#[test]
fn sliding_window_is_absolute_and_saturating() {
    assert_eq!(sliding_window_range(0, nonzero(4)), 0..0);
    assert_eq!(sliding_window_range(3, nonzero(4)), 0..3);
    assert_eq!(sliding_window_range(4, nonzero(4)), 0..4);
    assert_eq!(sliding_window_range(9, nonzero(4)), 5..9);
}

#[test]
fn write_plan_handles_empty_contiguous_wrapped_and_oversized_chunks() {
    let empty = plan_ring_write(3, 0, nonzero(4)).expect("valid plan");
    assert_eq!(empty.appended_positions(), 3..3);
    assert!(empty.segments().is_empty());

    let contiguous = plan_ring_write(1, 2, nonzero(4)).expect("valid plan");
    assert_eq!(contiguous.retained_input(), 0..2);
    assert_eq!(contiguous.segments()[0].input(), 0..2);
    assert_eq!(contiguous.segments()[0].physical(), 1..3);

    let wrapped = plan_ring_write(3, 3, nonzero(4)).expect("valid plan");
    assert_eq!(wrapped.segments().len(), 2);
    assert_eq!(wrapped.segments()[0].physical(), 3..4);
    assert_eq!(wrapped.segments()[1].physical(), 0..2);

    let oversized = plan_ring_write(2, 7, nonzero(4)).expect("valid plan");
    assert_eq!(oversized.appended_positions(), 2..9);
    assert_eq!(oversized.retained_input(), 3..7);
    assert_eq!(oversized.segments()[0].input(), 3..6);
    assert_eq!(oversized.segments()[0].physical(), 1..4);
    assert_eq!(oversized.segments()[1].input(), 6..7);
    assert_eq!(oversized.segments()[1].physical(), 0..1);
}

#[test]
fn staged_transaction_commits_only_an_accepted_prefix() {
    let staged = plan_staged_ring_transaction(6, 4).expect("valid staging plan");
    assert_eq!(staged.staged_positions(), 6..10);
    assert_eq!(staged.staged_input(), 0..4);

    let commit = staged
        .commit_prefix(2, nonzero(4))
        .expect("accepted prefix is staged");
    assert_eq!(commit.appended_positions(), 6..8);
    assert_eq!(commit.retained_input(), 0..2);
    assert_eq!(commit.segments()[0].physical(), 2..4);
    assert_eq!(
        staged.commit_prefix(5, nonzero(4)),
        Err(RingPlanError::CommitPrefixExceedsStaged {
            accepted: 5,
            staged: 4,
        })
    );
}

#[test]
fn read_plan_returns_chronological_slices() {
    let plan = plan_ring_read(10, nonzero(4), nonzero(4)).expect("valid plan");
    assert_eq!(plan.retained_positions(), 6..10);
    assert_eq!(plan.segments().len(), 2);
    assert_eq!(plan.segments()[0].absolute(), 6..8);
    assert_eq!(plan.segments()[0].physical(), 2..4);
    assert_eq!(plan.segments()[1].absolute(), 8..10);
    assert_eq!(plan.segments()[1].physical(), 0..2);

    let partial = plan_ring_read(2, nonzero(3), nonzero(8)).expect("valid plan");
    assert_eq!(partial.retained_positions(), 0..2);
    assert_eq!(partial.segments().len(), 1);
    assert_eq!(partial.segments()[0].physical(), 0..2);
}

#[test]
fn read_plan_rejects_impossible_window() {
    assert_eq!(
        plan_ring_read(4, nonzero(5), nonzero(4)),
        Err(RingPlanError::WindowExceedsCapacity {
            window: 5,
            capacity: 4,
        })
    );
}

#[test]
fn mirrored_ring_writes_both_halves_and_reads_one_slice() {
    let writes = plan_mirrored_ring_write(3, 3, nonzero(4)).expect("valid plan");
    assert_eq!(writes.len(), 2);
    assert_eq!(writes[0].input(), 0..1);
    assert_eq!(writes[0].primary(), 3..4);
    assert_eq!(writes[0].mirror(), 7..8);
    assert_eq!(writes[1].primary(), 0..2);
    assert_eq!(writes[1].mirror(), 4..6);

    let read = plan_mirrored_ring_read(6, nonzero(4)).expect("valid plan");
    assert_eq!(read.retained_positions(), 2..6);
    assert_eq!(read.physical(), 2..6);
}

#[test]
fn arithmetic_errors_are_reported() {
    assert_eq!(
        plan_ring_write(usize::MAX, 1, nonzero(4)),
        Err(RingPlanError::PositionOverflow {
            start_position: usize::MAX,
            token_count: 1,
        })
    );
    assert_eq!(
        plan_staged_ring_transaction(usize::MAX, 1),
        Err(RingPlanError::PositionOverflow {
            start_position: usize::MAX,
            token_count: 1,
        })
    );
    let too_large = nonzero(usize::MAX / 2 + 1);
    assert_eq!(
        plan_mirrored_ring_read(0, too_large),
        Err(RingPlanError::DoubledCapacityOverflow {
            capacity: usize::MAX / 2 + 1,
        })
    );
    assert_eq!(
        plan_mirrored_ring_write(0, 0, too_large),
        Err(RingPlanError::DoubledCapacityOverflow {
            capacity: usize::MAX / 2 + 1,
        })
    );
}

#[test]
fn error_messages_include_planning_context() {
    assert_eq!(
        RingPlanError::PositionOverflow {
            start_position: 10,
            token_count: 4,
        }
        .to_string(),
        "absolute position 10 overflows when appending 4 tokens"
    );
    assert_eq!(
        RingPlanError::WindowExceedsCapacity {
            window: 9,
            capacity: 8,
        }
        .to_string(),
        "sliding window 9 exceeds physical ring capacity 8"
    );
    assert_eq!(
        RingPlanError::DoubledCapacityOverflow { capacity: 12 }.to_string(),
        "mirrored ring capacity 2 * 12 overflows usize"
    );
    assert_eq!(
        RingPlanError::CommitPrefixExceedsStaged {
            accepted: 4,
            staged: 2,
        }
        .to_string(),
        "cannot commit 4 ring entries from a staging buffer of length 2"
    );
}

#[test]
fn exhaustive_write_plans_match_ring_simulation() {
    for capacity in 1..=8 {
        for start in 0..=20 {
            for count in 0..=20 {
                let plan = plan_ring_write(start, count, nonzero(capacity))
                    .expect("small positions do not overflow");
                let mut ring = vec![None; capacity];
                for input_index in 0..count {
                    ring[(start + input_index) % capacity] = Some(input_index);
                }

                let planned = plan
                    .segments()
                    .iter()
                    .flat_map(|segment| segment.input().zip(segment.physical()))
                    .collect::<Vec<_>>();
                let first_retained = count.saturating_sub(capacity);
                let expected = (first_retained..count)
                    .map(|input| (input, (start + input) % capacity))
                    .collect::<Vec<_>>();
                assert_eq!(planned, expected);
                for (input, slot) in planned {
                    assert_eq!(ring[slot], Some(input));
                }
            }
        }
    }
}

#[test]
fn exhaustive_read_plans_reconstruct_absolute_order() {
    for capacity in 1..=8 {
        for window in 1..=capacity {
            for length in 0..=24 {
                let plan = plan_ring_read(length, nonzero(window), nonzero(capacity))
                    .expect("window fits");
                let mappings = plan
                    .segments()
                    .iter()
                    .flat_map(|segment| segment.absolute().zip(segment.physical()))
                    .collect::<Vec<_>>();
                let expected = sliding_window_range(length, nonzero(window))
                    .map(|absolute| (absolute, absolute % capacity))
                    .collect::<Vec<_>>();
                assert_eq!(mappings, expected);
            }
        }
    }
}

#[test]
fn exhaustive_mirrored_reads_are_contiguous() {
    for capacity in 1..=16 {
        for length in 0..=64 {
            let plan = plan_mirrored_ring_read(length, nonzero(capacity))
                .expect("small capacity can be doubled");
            assert_eq!(
                plan.physical().len(),
                plan.retained_positions().len(),
                "capacity={capacity}, length={length}"
            );
            for (absolute, physical) in plan.retained_positions().zip(plan.physical()) {
                assert_eq!(physical % capacity, absolute % capacity);
            }
        }
    }
}

#[test]
fn exhaustive_staged_commits_preserve_rollback_and_visible_values() {
    for capacity in 1..=8 {
        for baseline_length in 0..=16 {
            let mut committed_ring = vec![None; capacity];
            for absolute in sliding_window_range(baseline_length, nonzero(capacity)) {
                committed_ring[absolute % capacity] = Some(absolute);
            }
            let unchanged_for_rollback = committed_ring.clone();

            for proposed in 0..=10 {
                let staged = plan_staged_ring_transaction(baseline_length, proposed)
                    .expect("small positions cannot overflow");
                // Proposals live outside the committed ring, so rollback is
                // exactly a drop and cannot corrupt wrapped baseline slots.
                assert_eq!(committed_ring, unchanged_for_rollback);

                for accepted in 0..=proposed {
                    let mut after_commit = committed_ring.clone();
                    let write = staged
                        .commit_prefix(accepted, nonzero(capacity))
                        .expect("accepted prefix is staged");
                    for segment in write.segments() {
                        for (input, slot) in segment.input().zip(segment.physical()) {
                            after_commit[slot] = Some(baseline_length + input);
                        }
                    }

                    let committed_length = baseline_length + accepted;
                    for absolute in sliding_window_range(committed_length, nonzero(capacity)) {
                        assert_eq!(
                            after_commit[absolute % capacity],
                            Some(absolute),
                            "capacity={capacity}, baseline={baseline_length}, proposed={proposed}, accepted={accepted}"
                        );
                    }
                }
            }
        }
    }
}
