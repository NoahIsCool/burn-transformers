//! Differential cache state machine with independent logical and ring-storage oracles.

use std::num::NonZeroUsize;
use transformers::cache::{
    CacheConfig, CacheMark, SequenceLengths, TransactionalCache, plan_ring_write,
    plan_staged_ring_transaction,
};

/// Checks physical ring writes against `slot = absolute_position mod capacity`.
fn check_ring(start: usize, count: usize, capacity: usize) {
    let capacity = NonZeroUsize::new(capacity).unwrap();
    let plan = plan_ring_write(start, count, capacity);
    if start.checked_add(count).is_none() {
        assert!(plan.is_err());
        return;
    }
    let plan = plan.unwrap();
    assert_eq!(plan.appended_positions(), start..start + count);
    let retained = count.saturating_sub(capacity.get())..count;
    assert_eq!(plan.retained_input(), retained.clone());
    let observed = plan
        .segments()
        .iter()
        .flat_map(|segment| segment.input().zip(segment.physical()))
        .collect::<Vec<_>>();
    let expected = retained
        .map(|offset| (offset, (start + offset) % capacity.get()))
        .collect::<Vec<_>>();
    assert_eq!(observed, expected);
    let staged = plan_staged_ring_transaction(start, count).unwrap();
    assert_eq!(staged.commit_prefix(count, capacity).unwrap(), plan);
    if count < usize::MAX {
        assert!(staged.commit_prefix(count + 1, capacity).is_err());
    }
}

/// Runs at most 128 mutations; rejection must preserve both lengths and transaction state.
pub fn exercise(input: &[u8]) {
    let mut bytes = crate::Bytes::new(input);
    let batch = usize::from(bytes.byte() % 4) + 1;
    let capacity = usize::from(bytes.byte() % 32) + 1;
    let mut state = SequenceLengths::new(CacheConfig::new(batch, capacity).unwrap());
    let mut expected = vec![0_usize; batch];
    let mut active: Option<(CacheMark, Vec<usize>)> = None;
    let mut stale: Option<CacheMark> = None;
    for _ in 0..input.len().min(512).div_ceil(4) {
        let operation = bytes.byte() % 10;
        let sequence = usize::from(bytes.byte() % 6);
        let value = bytes.size();
        let second = bytes.size();
        let previous = expected.clone();
        let was_active = active.is_some();
        let success = match operation {
            0 => {
                let next = expected
                    .get(sequence)
                    .and_then(|old| old.checked_add(value))
                    .filter(|&next| next <= capacity);
                let result = state.advance(sequence, value);
                assert_eq!(result.is_ok(), next.is_some());
                if let Some(next) = next {
                    expected[sequence] = next;
                }
                result.is_ok()
            }
            1 => {
                let counts = (0..batch)
                    .map(|index| if index == 0 { value } else { second })
                    .collect::<Vec<_>>();
                let next = expected
                    .iter()
                    .zip(&counts)
                    .map(|(&old, &count)| old.checked_add(count).filter(|&next| next <= capacity))
                    .collect::<Option<Vec<_>>>();
                let result = state.advance_all(&counts);
                assert_eq!(result.is_ok(), next.is_some());
                if let Some(next) = next {
                    expected = next;
                }
                result.is_ok()
            }
            2 => {
                let valid =
                    active.is_none() && expected.get(sequence).is_some_and(|&old| value <= old);
                let result = state.truncate(sequence, value);
                assert_eq!(result.is_ok(), valid);
                if valid {
                    expected[sequence] = value;
                }
                valid
            }
            3 => {
                let lengths = vec![value; batch];
                let valid = active.is_none() && expected.iter().all(|&old| value <= old);
                assert_eq!(state.truncate_all(&lengths).is_ok(), valid);
                if valid {
                    expected = lengths;
                }
                valid
            }
            4 => {
                let indices = (0..batch)
                    .map(|index| {
                        if index == 0 {
                            sequence
                        } else {
                            second % (batch + 1)
                        }
                    })
                    .collect::<Vec<_>>();
                let valid = active.is_none() && indices.iter().all(|&index| index < batch);
                assert_eq!(state.reorder(&indices).is_ok(), valid);
                if valid {
                    expected = indices.iter().map(|&index| previous[index]).collect();
                }
                valid
            }
            5 => {
                let valid = active.is_none();
                assert_eq!(state.reset().is_ok(), valid);
                if valid {
                    expected.fill(0);
                }
                valid
            }
            6 => {
                let result = state.mark();
                assert_eq!(result.is_ok(), active.is_none());
                if let Ok(mark) = result {
                    active = Some((mark, expected.clone()));
                    true
                } else {
                    false
                }
            }
            7 => {
                if let Some((mark, baseline)) = &active {
                    let valid = expected
                        .iter()
                        .zip(baseline)
                        .all(|(&now, &old)| value <= now - old);
                    assert_eq!(state.commit_prefix(mark.clone(), value).is_ok(), valid);
                    if valid {
                        expected = baseline.iter().map(|&old| old + value).collect();
                        stale = Some(mark.clone());
                        active = None;
                    }
                    valid
                } else {
                    false
                }
            }
            8 => {
                if let Some((mark, baseline)) = active.take() {
                    state.rollback(mark.clone()).unwrap();
                    expected = baseline;
                    stale = Some(mark);
                    true
                } else {
                    false
                }
            }
            _ => {
                if let Some(mark) = &stale {
                    assert!(state.rollback(mark.clone()).is_err());
                }
                false
            }
        };
        // `expected` follows the working cursor; committed lengths stay at the
        // transaction baseline until a commit.
        assert_eq!(state.working_lengths(), expected);
        let committed = active.as_ref().map_or(&expected, |(_, baseline)| baseline);
        assert_eq!(state.lengths(), committed.as_slice());
        assert_eq!(state.has_active_transaction(), active.is_some());
        assert!(expected.iter().all(|&length| length <= capacity));
        if !success {
            assert_eq!(expected, previous);
            assert_eq!(state.has_active_transaction(), was_active);
        }
        check_ring(value, second, capacity);
    }
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
