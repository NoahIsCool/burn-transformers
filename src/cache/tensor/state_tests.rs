//! Multi-row publication boundaries for coordinated main and auxiliary cache state.

use burn::backend::Flex;
use burn::tensor::{DType, Tensor};

use super::*;
use crate::cache::KvCacheStorage;

/// Deterministic host tensors for exact lifecycle-state comparisons.
type TestBackend = Flex<f32, i32>;

/// Allocates two rows with a full main cache and a two-token auxiliary ring.
fn paired_state() -> TensorDecodeState<TestBackend> {
    let config = CacheConfig::new(2, 8)
        .expect("valid config")
        .with_sliding_window(2)
        .expect("valid window");
    let mut state =
        TensorDecodeState::new(config, 2, &Default::default()).expect("state allocation");
    let full =
        NativeKvCacheSpec::new(1, 1, 1, KvCacheStorage::Full, DType::F32).expect("full geometry");
    let ring = NativeKvCacheSpec::new(1, 1, 1, KvCacheStorage::SlidingRing, DType::F32)
        .expect("ring geometry");
    state.main_mut().add_native_kv(full).expect("main cache");
    state
        .enable_mtp()
        .expect("MTP bank")
        .add_native_kv(ring)
        .expect("MTP cache");
    for sequence in 0..2 {
        let value = (sequence + 1) as f32;
        let keys = Tensor::full([1, 1, 3, 1], value, &Default::default());
        let values = Tensor::full([1, 1, 3, 1], value * 10.0, &Default::default());
        state
            .main_mut()
            .native_kv_mut(0)
            .expect("main cache")
            .write_sequence(0, sequence, 0, keys.clone(), values.clone())
            .expect("main baseline");
        state
            .mtp_mut()
            .expect("MTP bank")
            .native_kv_mut(0)
            .expect("MTP cache")
            .write_sequence(0, sequence, 0, keys, values)
            .expect("auxiliary baseline");
    }
    state.advance(&[3, 3]).expect("main lengths");
    state.advance_mtp(&[3, 3]).expect("auxiliary lengths");
    state
}

/// An unsafe final auxiliary row cannot clear any earlier row or any main tensor.
#[test]
fn auxiliary_second_row_rewind_failure_preserves_both_banks() {
    let mut state = paired_state();
    assert!(matches!(
        state.truncate_prefixes(&[0, 0], &[0, 2]),
        Err(TensorCacheError::UnsafeWrappedTruncate { sequence: 1, .. })
    ));
    assert_eq!(state.lengths(), &[3, 3]);
    assert_eq!(state.mtp_lengths(), Some([3, 3].as_slice()));
    for bank in [state.main(), state.mtp().expect("MTP bank")] {
        let cache = bank.native_kv(0).expect("cache group");
        for sequence in 0..2 {
            let read = cache.read_sequence(0, sequence, 3).expect("committed row");
            let (keys, values) = read.materialize().expect("nonempty row");
            let retained = keys.dims()[2];
            let expected = (sequence + 1) as f32;
            assert_eq!(
                keys.into_data().to_vec::<f32>().expect("F32 keys"),
                vec![expected; retained]
            );
            assert_eq!(
                values.into_data().to_vec::<f32>().expect("F32 values"),
                vec![expected * 10.0; retained]
            );
        }
    }
}

/// Batch-size-free constructors match the explicit ones.
#[test]
fn config_only_constructors_derive_the_batch_size() {
    let config = CacheConfig::new(3, 5).expect("valid config");
    let state = TensorDecodeState::<TestBackend>::with_config(config, &Default::default());
    assert_eq!(state.batch_size(), 3);
    assert_eq!(state.lengths(), &[0, 0, 0]);
    let bank = TensorCacheBank::<TestBackend>::with_config(config, &Default::default());
    assert_eq!(bank.batch_size(), 3);
    assert_eq!(bank.config(), config);
}

/// The public metadata view reflects the tensor transaction instead of hiding it.
#[test]
fn metadata_view_reports_the_tensor_transaction() {
    let config = CacheConfig::new(2, 8).expect("valid config");
    let mut state = TensorDecodeState::<TestBackend>::with_config(config, &Default::default());
    let spec = NativeKvCacheSpec::new(1, 1, 1, KvCacheStorage::Full, DType::F32)
        .expect("geometry")
        .with_staging_capacity(2);
    state.main_mut().add_native_kv(spec).expect("cache");
    state.advance(&[3, 1]).expect("prefill");
    let mark = state.begin_transaction().expect("transaction");
    assert!(state.metadata().has_active_transaction());
    assert_eq!(mark.baseline_lengths(), &[3, 1]);
    assert_eq!(mark.mtp_baseline_lengths(), None);
    state.advance(&[1, 2]).expect("proposal");
    assert_eq!(state.metadata().lengths(), &[3, 1]);
    assert_eq!(state.metadata().working_lengths(), &[4, 3]);
    assert_eq!(state.working_lengths(), &[4, 3]);
    state.rollback(mark).expect("rollback");
    assert!(!state.metadata().has_active_transaction());
    assert_eq!(state.working_lengths(), &[3, 1]);
}

/// Groups cannot be added once lengths are non-zero, so they never expose zero "history".
#[test]
fn groups_are_refused_after_tokens_are_committed() {
    let config = CacheConfig::new(2, 8).expect("valid config");
    let mut state = TensorDecodeState::<TestBackend>::with_config(config, &Default::default());
    let kv = NativeKvCacheSpec::new(1, 1, 1, KvCacheStorage::Full, DType::F32).expect("geometry");
    let recurrent =
        crate::cache::RecurrentStateSpec::new(1, 1, 1, 1, 1, 1, DType::F32).expect("geometry");
    state
        .main_mut()
        .add_native_kv(kv)
        .expect("empty state accepts groups");
    state.advance(&[0, 2]).expect("prefill one row");
    assert!(matches!(
        state.main_mut().add_native_kv(kv),
        Err(TensorCacheError::TopologyAfterHistory { .. })
    ));
    assert!(matches!(
        state.main_mut().add_recurrent(recurrent),
        Err(TensorCacheError::TopologyAfterHistory { .. })
    ));
    assert!(
        TensorCacheError::TopologyAfterHistory {
            operation: "add native KV cache"
        }
        .to_string()
        .contains("zero-filled history")
    );

    // The auxiliary bank has its own, still empty, cursor.
    state
        .enable_mtp()
        .expect("MTP bank")
        .add_native_kv(kv)
        .expect("MTP cursor is empty");
    state.advance_mtp(&[1, 0]).expect("MTP prefill");
    assert!(matches!(
        state.mtp_mut().expect("MTP bank").add_native_kv(kv),
        Err(TensorCacheError::TopologyAfterHistory { .. })
    ));

    // Duplicating the empty row empties the main bank; reset empties both.
    state.reorder(&[0, 0]).expect("beam reorder");
    assert_eq!(state.lengths(), &[0, 0]);
    state
        .main_mut()
        .add_recurrent(recurrent)
        .expect("main is empty again");
    state.advance(&[1, 1]).expect("prefill");
    state.reset().expect("reset");
    state
        .main_mut()
        .add_native_kv(kv)
        .expect("reset unlocks topology");
    state
        .mtp_mut()
        .expect("MTP bank")
        .add_native_kv(kv)
        .expect("reset unlocks the auxiliary bank");
}
