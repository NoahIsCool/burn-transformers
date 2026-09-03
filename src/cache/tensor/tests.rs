use burn::backend::Flex;
use burn::tensor::{DType, Tensor, TensorData};

use crate::architecture::DecodeState as ArchitectureDecodeState;
use crate::cache::CacheConfig;

use super::*;

type TestBackend = Flex<f32, i32>;

fn device() -> <TestBackend as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

fn tensor3(values: Vec<f32>, shape: [usize; 3]) -> Tensor<TestBackend, 3> {
    Tensor::from_data(TensorData::new(values, shape), &device())
}

fn tensor4(values: Vec<f32>, shape: [usize; 4]) -> Tensor<TestBackend, 4> {
    Tensor::from_data(TensorData::new(values, shape), &device())
}

fn tensor5(values: Vec<f32>, shape: [usize; 5]) -> Tensor<TestBackend, 5> {
    Tensor::from_data(TensorData::new(values, shape), &device())
}

fn scalar_kv(values: &[f32]) -> (Tensor<TestBackend, 4>, Tensor<TestBackend, 4>) {
    let shape = [1, 1, values.len(), 1];
    (
        tensor4(values.to_vec(), shape),
        tensor4(values.iter().map(|value| value * 10.0).collect(), shape),
    )
}

fn materialized_keys(
    cache: &NativeKvCache<TestBackend>,
    layer: usize,
    sequence: usize,
    length: usize,
) -> Vec<f32> {
    cache
        .read_sequence(layer, sequence, length)
        .expect("valid cache read")
        .materialize()
        .expect("non-empty cache read")
        .0
        .into_data()
        .to_vec::<f32>()
        .expect("f32 cache")
}

fn scalar_kv_spec(storage: KvCacheStorage, staging: usize) -> NativeKvCacheSpec {
    NativeKvCacheSpec::new(1, 1, 1, storage, DType::F32)
        .expect("valid KV spec")
        .with_staging_capacity(staging)
}

fn scalar_recurrent_spec(staging: usize) -> RecurrentStateSpec {
    RecurrentStateSpec::new(1, 1, 1, 1, 1, 2, DType::F32)
        .expect("valid recurrent spec")
        .with_staging_capacity(staging)
}

#[test]
fn specs_validate_dimensions_dtype_and_accessors() {
    for (field, args) in [
        ("layers", [0, 1, 1]),
        ("key_value_heads", [1, 0, 1]),
        ("head_dimension", [1, 1, 0]),
    ] {
        assert!(matches!(
            NativeKvCacheSpec::new(
                args[0],
                args[1],
                args[2],
                KvCacheStorage::Full,
                DType::F32
            ),
            Err(TensorCacheError::ZeroDimension { field: actual }) if actual == field
        ));
    }
    assert!(matches!(
        NativeKvCacheSpec::new(1, 1, 1, KvCacheStorage::Full, DType::I32),
        Err(TensorCacheError::NonFloatDType { dtype: DType::I32 })
    ));

    let kv = NativeKvCacheSpec::new(3, 4, 8, KvCacheStorage::MirroredSlidingRing, DType::F16)
        .expect("valid KV spec")
        .with_staging_capacity(6);
    assert_eq!(kv.layers(), 3);
    assert_eq!(kv.key_value_heads(), 4);
    assert_eq!(kv.head_dimension(), 8);
    assert_eq!(kv.storage(), KvCacheStorage::MirroredSlidingRing);
    assert_eq!(kv.storage().as_str(), "mirrored sliding ring");
    assert_eq!(KvCacheStorage::Full.as_str(), "full");
    assert_eq!(KvCacheStorage::SlidingRing.as_str(), "sliding ring");
    assert_eq!(kv.dtype(), DType::F16);
    assert_eq!(kv.staging_capacity(), 6);

    let recurrent = RecurrentStateSpec::new(2, 3, 4, 5, 6, 7, DType::BF16)
        .expect("valid recurrent spec")
        .with_staging_capacity(8);
    assert_eq!(recurrent.layers(), 2);
    assert_eq!(recurrent.state_heads(), 3);
    assert_eq!(recurrent.state_rows(), 4);
    assert_eq!(recurrent.state_columns(), 5);
    assert_eq!(recurrent.convolution_channels(), 6);
    assert_eq!(recurrent.convolution_width(), 7);
    assert_eq!(recurrent.dtype(), DType::BF16);
    assert_eq!(recurrent.staging_capacity(), 8);
    assert_eq!(recurrent.matrix_shape(9), [9, 3, 4, 5]);
    assert_eq!(recurrent.convolution_shape(9), [9, 6, 7]);

    for field in [
        "layers",
        "state_heads",
        "state_rows",
        "state_columns",
        "convolution_channels",
        "convolution_width",
    ] {
        let mut dims = [1; 6];
        let index = match field {
            "layers" => 0,
            "state_heads" => 1,
            "state_rows" => 2,
            "state_columns" => 3,
            "convolution_channels" => 4,
            _ => 5,
        };
        dims[index] = 0;
        assert!(matches!(
            RecurrentStateSpec::new(
                dims[0],
                dims[1],
                dims[2],
                dims[3],
                dims[4],
                dims[5],
                DType::F32
            ),
            Err(TensorCacheError::ZeroDimension { field: actual }) if actual == field
        ));
    }
    assert!(matches!(
        RecurrentStateSpec::new(1, 1, 1, 1, 1, 1, DType::I32),
        Err(TensorCacheError::NonFloatDType { dtype: DType::I32 })
    ));
}

#[test]
fn allocation_shapes_match_full_ring_and_mirrored_layouts() {
    let device = device();
    let global = CacheConfig::new(2, 5).expect("valid config");
    let full = NativeKvCache::<TestBackend>::new(
        global,
        2,
        NativeKvCacheSpec::new(2, 3, 4, KvCacheStorage::Full, DType::F32).expect("valid spec"),
        &device,
    )
    .expect("valid cache");
    assert_eq!(full.committed_time_slots(), 5);
    assert_eq!(full.keys(0).expect("layer").dims(), [2, 3, 5, 4]);
    assert_eq!(full.values(1).expect("layer").dims(), [2, 3, 5, 4]);
    assert_eq!(full.config(), global);
    assert_eq!(full.device(), &device);

    assert!(matches!(
        NativeKvCache::<TestBackend>::new(
            global,
            2,
            scalar_kv_spec(KvCacheStorage::SlidingRing, 0),
            &device
        ),
        Err(TensorCacheError::SlidingWindowRequired { .. })
    ));
    let local = global.with_sliding_window(2).expect("valid window");
    let ring = NativeKvCache::<TestBackend>::new(
        local,
        2,
        scalar_kv_spec(KvCacheStorage::SlidingRing, 0),
        &device,
    )
    .expect("ring cache");
    let mirrored = NativeKvCache::<TestBackend>::new(
        local,
        2,
        scalar_kv_spec(KvCacheStorage::MirroredSlidingRing, 0),
        &device,
    )
    .expect("mirrored cache");
    assert_eq!(ring.committed_time_slots(), 2);
    assert_eq!(mirrored.committed_time_slots(), 4);
}

#[test]
fn full_cache_writes_reads_and_checks_every_tensor_boundary() {
    let device = device();
    let config = CacheConfig::new(2, 5).expect("valid config");
    let mut cache = NativeKvCache::<TestBackend>::new(
        config,
        2,
        scalar_kv_spec(KvCacheStorage::Full, 0),
        &device,
    )
    .expect("full cache");
    let (keys, values) = scalar_kv(&[1.0, 2.0, 3.0]);
    cache
        .write_sequence(0, 0, 1, keys, values)
        .expect("in-bounds write");
    assert_eq!(materialized_keys(&cache, 0, 0, 4), [0.0, 1.0, 2.0, 3.0]);
    assert_eq!(
        cache
            .read_sequence(0, 0, 0)
            .expect("empty read")
            .segments()
            .len(),
        0
    );
    assert!(matches!(
        cache.read_sequence(0, 0, 6),
        Err(TensorCacheError::ReadExceedsCapacity { .. })
    ));

    let (keys, values) = scalar_kv(&[4.0, 5.0]);
    assert!(matches!(
        cache.write_sequence(0, 0, 4, keys, values),
        Err(TensorCacheError::WriteExceedsCapacity { .. })
    ));
    let (keys, values) = scalar_kv(&[1.0]);
    assert!(matches!(
        cache.write_sequence(1, 0, 0, keys, values),
        Err(TensorCacheError::LayerOutOfBounds { .. })
    ));
    let (keys, values) = scalar_kv(&[1.0]);
    assert!(matches!(
        cache.write_sequence(0, 2, 0, keys, values),
        Err(TensorCacheError::SequenceOutOfBounds { .. })
    ));

    let bad_keys = tensor4(vec![1.0, 2.0], [1, 2, 1, 1]);
    let (_, values) = scalar_kv(&[1.0]);
    assert!(matches!(
        cache.write_sequence(0, 0, 0, bad_keys, values),
        Err(TensorCacheError::ShapeMismatch { tensor: "keys", .. })
    ));
    let f64_keys = Tensor::<TestBackend, 4>::zeros([1, 1, 1, 1], (&device, DType::F64));
    let (_, values) = scalar_kv(&[1.0]);
    assert!(matches!(
        cache.write_sequence(0, 0, 0, f64_keys, values),
        Err(TensorCacheError::DTypeMismatch { tensor: "keys", .. })
    ));
}

#[test]
fn ring_and_mirrored_cache_reads_preserve_chronology() {
    let device = device();
    let config = CacheConfig::new(1, 8)
        .expect("valid config")
        .with_sliding_window(3)
        .expect("valid window");
    for storage in [
        KvCacheStorage::SlidingRing,
        KvCacheStorage::MirroredSlidingRing,
    ] {
        let mut cache =
            NativeKvCache::<TestBackend>::new(config, 1, scalar_kv_spec(storage, 0), &device)
                .expect("ring cache");
        let (keys, values) = scalar_kv(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        cache
            .write_sequence(0, 0, 0, keys, values)
            .expect("oversized ring write");
        let read = cache.read_sequence(0, 0, 5).expect("valid read");
        assert_eq!(read.retained_positions(), 2..5);
        assert_eq!(
            read.segments().len(),
            if storage == KvCacheStorage::SlidingRing {
                2
            } else {
                1
            }
        );
        assert_eq!(
            read.segments()
                .iter()
                .flat_map(|segment| segment.absolute_positions())
                .collect::<Vec<_>>(),
            [2, 3, 4]
        );
        assert_eq!(materialized_keys(&cache, 0, 0, 5), [3.0, 4.0, 5.0]);
    }
}

#[test]
fn speculative_ring_rollback_never_overwrites_committed_slots() {
    let device = device();
    let config = CacheConfig::new(1, 10)
        .expect("valid config")
        .with_sliding_window(4)
        .expect("valid window");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::SlidingRing, 4))
        .expect("KV cache");
    let (keys, values) = scalar_kv(&[0.0, 1.0, 2.0, 3.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .write_sequence(0, 0, 0, keys, values)
        .expect("baseline write");
    state.advance(&[4]).expect("baseline length");

    let mark = state.begin_transaction().expect("transaction");
    assert_eq!(mark.baseline_lengths(), &[4]);
    let (keys, values) = scalar_kv(&[10.0, 11.0, 12.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .stage_sequence(0, 0, keys, values)
        .expect("stage proposal");
    state.advance(&[3]).expect("proposal length");
    state.rollback(mark).expect("rollback");
    assert_eq!(state.metadata().lengths(), &[4]);
    assert_eq!(
        materialized_keys(state.main().native_kv(0).expect("cache"), 0, 0, 4),
        [0.0, 1.0, 2.0, 3.0]
    );

    let mark = state.begin_transaction().expect("transaction");
    let stale = mark.clone();
    let (keys, values) = scalar_kv(&[10.0, 11.0, 12.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .stage_sequence(0, 0, keys, values)
        .expect("stage proposal");
    state.advance(&[3]).expect("proposal length");
    state.commit_prefix(mark, &[2]).expect("commit prefix");
    assert_eq!(state.metadata().lengths(), &[6]);
    assert_eq!(
        materialized_keys(state.main().native_kv(0).expect("cache"), 0, 0, 6),
        [2.0, 3.0, 10.0, 11.0]
    );
    assert!(matches!(
        state.rollback(stale),
        Err(TensorCacheError::TransactionNotActive)
    ));
}

#[test]
fn incomplete_or_oversized_staging_is_atomic_and_recoverable() {
    let device = device();
    let config = CacheConfig::new(1, 8).expect("valid config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    let spec = NativeKvCacheSpec::new(2, 1, 1, KvCacheStorage::Full, DType::F32)
        .expect("spec")
        .with_staging_capacity(2);
    state.main_mut().add_native_kv(spec).expect("cache");

    let mark = state.begin_transaction().expect("transaction");
    let rollback = mark.clone();
    let (keys, values) = scalar_kv(&[1.0, 2.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .stage_sequence(0, 0, keys, values)
        .expect("one layer staged");
    state.advance(&[2]).expect("proposal");
    assert!(matches!(
        state.commit_prefix(mark, &[1]),
        Err(TensorCacheError::IncompleteStaging {
            layer: 1,
            expected: 2,
            actual: 0,
            ..
        })
    ));
    assert!(state.has_active_transaction());
    state.rollback(rollback).expect("recoverable rollback");

    let _mark = state.begin_transaction().expect("transaction");
    let (keys, values) = scalar_kv(&[1.0, 2.0, 3.0]);
    assert!(matches!(
        state
            .main_mut()
            .native_kv_mut(0)
            .expect("cache")
            .stage_sequence(0, 0, keys, values),
        Err(TensorCacheError::StagingCapacityExceeded {
            requested: 3,
            capacity: 2
        })
    ));
}

#[test]
fn recurrent_transactions_select_the_accepted_terminal_snapshot() {
    let device = device();
    let config = CacheConfig::new(1, 8).expect("valid config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_recurrent(scalar_recurrent_spec(3))
        .expect("recurrent cache");
    state
        .main_mut()
        .recurrent_mut(0)
        .expect("cache")
        .set_sequence_state(
            0,
            0,
            tensor4(vec![1.0], [1, 1, 1, 1]),
            tensor3(vec![2.0, 3.0], [1, 1, 2]),
        )
        .expect("baseline state");
    state.advance(&[1]).expect("baseline length");

    let mark = state.begin_transaction().expect("transaction");
    state
        .main_mut()
        .recurrent_mut(0)
        .expect("cache")
        .stage_sequence_snapshots(
            0,
            0,
            tensor5(vec![10.0, 20.0], [1, 2, 1, 1, 1]),
            tensor4(vec![11.0, 12.0, 21.0, 22.0], [1, 2, 1, 2]),
        )
        .expect("snapshots");
    state.advance(&[2]).expect("proposal");
    state.commit_prefix(mark, &[1]).expect("accept one");
    let recurrent = state.main().recurrent(0).expect("cache");
    assert_eq!(
        recurrent
            .matrix(0)
            .expect("matrix")
            .to_data()
            .to_vec::<f32>()
            .expect("f32"),
        [10.0]
    );
    assert_eq!(
        recurrent
            .convolution(0)
            .expect("convolution")
            .to_data()
            .to_vec::<f32>()
            .expect("f32"),
        [11.0, 12.0]
    );

    let mark = state.begin_transaction().expect("transaction");
    state
        .main_mut()
        .recurrent_mut(0)
        .expect("cache")
        .stage_sequence_snapshots(
            0,
            0,
            tensor5(vec![30.0], [1, 1, 1, 1, 1]),
            tensor4(vec![31.0, 32.0], [1, 1, 1, 2]),
        )
        .expect("snapshot");
    state.advance(&[1]).expect("proposal");
    state.rollback(mark).expect("rollback");
    assert_eq!(
        state
            .main()
            .recurrent(0)
            .expect("cache")
            .matrix(0)
            .expect("matrix")
            .to_data()
            .to_vec::<f32>()
            .expect("f32"),
        [10.0]
    );
}

#[test]
fn reorder_duplicates_main_recurrent_and_separate_mtp_rows() {
    let device = device();
    let config = CacheConfig::new(2, 4).expect("valid config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 2, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 0))
        .expect("main KV");
    state
        .main_mut()
        .add_recurrent(scalar_recurrent_spec(0))
        .expect("main recurrent");
    state
        .enable_mtp()
        .expect("MTP bank")
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 0))
        .expect("MTP KV");

    for (sequence, main_value, mtp_value, recurrent_value) in
        [(0, 1.0, 10.0, 3.0), (1, 2.0, 20.0, 4.0)]
    {
        let (keys, values) = scalar_kv(&[main_value]);
        state
            .main_mut()
            .native_kv_mut(0)
            .expect("main KV")
            .write_sequence(0, sequence, 0, keys, values)
            .expect("main write");
        let (keys, values) = scalar_kv(&[mtp_value]);
        state
            .mtp_mut()
            .expect("MTP")
            .native_kv_mut(0)
            .expect("MTP KV")
            .write_sequence(0, sequence, 0, keys, values)
            .expect("MTP write");
        state
            .main_mut()
            .recurrent_mut(0)
            .expect("recurrent")
            .set_sequence_state(
                0,
                sequence,
                tensor4(vec![recurrent_value], [1, 1, 1, 1]),
                tensor3(vec![recurrent_value; 2], [1, 1, 2]),
            )
            .expect("recurrent write");
    }
    state.advance(&[1, 1]).expect("lengths");
    state.reorder(&[1, 1]).expect("beam duplication");
    assert_eq!(state.metadata().lengths(), &[1, 1]);
    for sequence in 0..2 {
        assert_eq!(
            materialized_keys(state.main().native_kv(0).expect("cache"), 0, sequence, 1),
            [2.0]
        );
        assert_eq!(
            materialized_keys(
                state.mtp().expect("MTP").native_kv(0).expect("cache"),
                0,
                sequence,
                1
            ),
            [20.0]
        );
    }
    assert_eq!(
        state
            .main()
            .recurrent(0)
            .expect("cache")
            .matrix(0)
            .expect("matrix")
            .to_data()
            .to_vec::<f32>()
            .expect("f32"),
        [4.0, 4.0]
    );
    assert_eq!(state.device(), &device);
    assert_eq!(state.main().device(), &device);
    assert_eq!(state.mtp().expect("MTP").device(), &device);
}

#[test]
fn reset_clears_every_tensor_and_satisfies_architecture_contract() {
    let device = device();
    let config = CacheConfig::new(1, 4).expect("valid config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 0))
        .expect("cache");
    let (keys, values) = scalar_kv(&[7.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .write_sequence(0, 0, 0, keys, values)
        .expect("write");
    state.advance(&[1]).expect("length");

    assert_eq!(
        ArchitectureDecodeState::<TestBackend>::batch_size(&state),
        1
    );
    assert_eq!(ArchitectureDecodeState::<TestBackend>::capacity(&state), 4);
    assert_eq!(
        ArchitectureDecodeState::<TestBackend>::lengths(&state),
        &[1]
    );
    assert_eq!(
        ArchitectureDecodeState::<TestBackend>::device(&state),
        &device
    );
    ArchitectureDecodeState::<TestBackend>::reset(&mut state).expect("contract reset");
    assert_eq!(state.metadata().lengths(), &[0]);
    assert!(
        state
            .main()
            .native_kv(0)
            .expect("cache")
            .keys(0)
            .expect("keys")
            .to_data()
            .to_vec::<f32>()
            .expect("f32")
            .iter()
            .all(|&value| value == 0.0)
    );
}

#[test]
fn truncate_accepts_safe_prefixes_and_rejects_irreversible_rewinds() {
    let device = device();
    let full_config = CacheConfig::new(1, 8).expect("config");
    let mut full = TensorDecodeState::<TestBackend>::new(full_config, 1, &device).expect("state");
    full.main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 0))
        .expect("cache");
    full.advance(&[5]).expect("length");
    full.truncate(&[2]).expect("full prefix retained");
    assert_eq!(full.metadata().lengths(), &[2]);

    let ring_config = CacheConfig::new(1, 8)
        .expect("config")
        .with_sliding_window(3)
        .expect("window");
    let mut ring = TensorDecodeState::<TestBackend>::new(ring_config, 1, &device).expect("state");
    ring.main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::SlidingRing, 0))
        .expect("cache");
    ring.advance(&[3]).expect("unwrapped length");
    ring.truncate(&[2]).expect("unwrapped rewind");
    ring.advance(&[2]).expect("wrapped length");
    assert!(matches!(
        ring.truncate(&[3]),
        Err(TensorCacheError::UnsafeWrappedTruncate {
            current: 4,
            requested: 3,
            ..
        })
    ));
    ring.truncate(&[0]).expect("zero reset is exact");
    assert_eq!(ring.metadata().lengths(), &[0]);

    let mut recurrent =
        TensorDecodeState::<TestBackend>::new(full_config, 1, &device).expect("state");
    recurrent
        .main_mut()
        .add_recurrent(scalar_recurrent_spec(0))
        .expect("cache");
    recurrent.advance(&[2]).expect("length");
    assert!(matches!(
        recurrent.truncate(&[1]),
        Err(TensorCacheError::UnsafeRecurrentTruncate {
            current: 2,
            requested: 1,
            ..
        })
    ));
    recurrent.truncate(&[0]).expect("initial state restorable");
}

#[test]
fn lifecycle_validates_batch_capacity_transactions_and_bank_topology() {
    let device = device();
    let config = CacheConfig::new(2, 3).expect("config");
    assert!(matches!(
        TensorDecodeState::<TestBackend>::new(config, 1, &device),
        Err(TensorCacheError::BatchSizeMismatch {
            configured: 2,
            requested: 1
        })
    ));
    let mut state = TensorDecodeState::<TestBackend>::new(config, 2, &device).expect("state");
    assert!(matches!(
        state.advance(&[4, 0]),
        Err(TensorCacheError::Metadata(_))
    ));
    assert_eq!(state.metadata().lengths(), &[0, 0]);
    assert!(matches!(
        state.reorder(&[0]),
        Err(TensorCacheError::BatchSizeMismatch { .. })
    ));
    assert!(matches!(
        state.truncate(&[1, 0]),
        Err(TensorCacheError::Metadata(_))
    ));

    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 0))
        .expect("cache");
    assert!(matches!(
        state.begin_transaction(),
        Err(TensorCacheError::StagingDisabled)
    ));
    assert!(!state.has_active_transaction());
    assert!(matches!(
        state.main().native_kv(1),
        Err(TensorCacheError::CacheOutOfBounds { .. })
    ));
    assert!(matches!(
        state.main().recurrent(0),
        Err(TensorCacheError::CacheOutOfBounds { .. })
    ));

    let different = CacheConfig::new(2, 4).expect("different config");
    let bank = TensorCacheBank::<TestBackend>::new(different, 2, &device).expect("bank");
    assert!(matches!(
        state.attach_mtp(bank),
        Err(TensorCacheError::IncompatibleBank { bank: "MTP", .. })
    ));
    state.enable_mtp().expect("MTP");
    assert!(matches!(
        state.enable_mtp(),
        Err(TensorCacheError::MtpAlreadyEnabled)
    ));
}

#[test]
fn active_transactions_block_topology_and_lifecycle_changes() {
    let device = device();
    let config = CacheConfig::new(1, 4).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 2))
        .expect("cache");
    let mark = state.begin_transaction().expect("transaction");
    assert!(matches!(
        state.begin_transaction(),
        Err(TensorCacheError::TransactionAlreadyActive)
    ));
    assert!(matches!(
        state.reset(),
        Err(TensorCacheError::TransactionActive { operation: "reset" })
    ));
    assert!(matches!(
        state.reorder(&[0]),
        Err(TensorCacheError::TransactionActive {
            operation: "reorder"
        })
    ));
    assert!(matches!(
        state.truncate(&[0]),
        Err(TensorCacheError::TransactionActive {
            operation: "truncate"
        })
    ));
    assert!(matches!(
        state.main_mut().add_recurrent(scalar_recurrent_spec(1)),
        Err(TensorCacheError::TransactionActive { .. })
    ));
    let (keys, values) = scalar_kv(&[1.0]);
    assert!(matches!(
        state
            .main_mut()
            .native_kv_mut(0)
            .expect("cache")
            .write_sequence(0, 0, 0, keys, values),
        Err(TensorCacheError::TransactionActive { .. })
    ));
    state.rollback(mark).expect("rollback");
}
