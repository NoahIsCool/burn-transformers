//! Tensor-cache allocation, numerical storage parity, and coordinated lifecycle contracts.

use burn::backend::Flex;
use burn::tensor::{DType, Tensor, TensorData};

use crate::architecture::DecodeState as ArchitectureDecodeState;
use crate::cache::CacheConfig;

use super::*;

/// Deterministic Flex backend with F32 values and portable I32 indices.
type TestBackend = Flex<f32, i32>;

/// Returns the deterministic Flex host device used by every tensor fixture.
fn device() -> <TestBackend as burn::tensor::backend::BackendTypes>::Device {
    Default::default()
}

/// Creates a rank-three F32 fixture with explicit values and shape.
fn tensor3(values: Vec<f32>, shape: [usize; 3]) -> Tensor<TestBackend, 3> {
    Tensor::from_data(TensorData::new(values, shape), &device())
}

/// Creates a rank-four F32 fixture with explicit values and shape.
fn tensor4(values: Vec<f32>, shape: [usize; 4]) -> Tensor<TestBackend, 4> {
    Tensor::from_data(TensorData::new(values, shape), &device())
}

/// Creates a rank-five F32 fixture with explicit values and shape.
fn tensor5(values: Vec<f32>, shape: [usize; 5]) -> Tensor<TestBackend, 5> {
    Tensor::from_data(TensorData::new(values, shape), &device())
}

/// Creates scalar-head keys and values ten times larger to detect pair mix-ups.
fn scalar_kv(values: &[f32]) -> (Tensor<TestBackend, 4>, Tensor<TestBackend, 4>) {
    let shape = [1, 1, values.len(), 1];
    (
        tensor4(values.to_vec(), shape),
        tensor4(values.iter().map(|value| value * 10.0).collect(), shape),
    )
}

/// Reads committed keys in chronological order and transfers them for exact assertions.
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

/// Reads committed and proposed keys in chronological order for exact assertions.
fn working_keys(
    cache: &NativeKvCache<TestBackend>,
    layer: usize,
    sequence: usize,
    committed_length: usize,
) -> Vec<f32> {
    cache
        .read_working_sequence(layer, sequence, committed_length)
        .expect("valid working cache read")
        .materialize()
        .expect("non-empty working cache read")
        .0
        .into_data()
        .to_vec::<f32>()
        .expect("f32 cache")
}

/// Uploads validated beam-source rows as the shared device selector.
fn selection(indices: &[usize]) -> Tensor<TestBackend, 1, burn::tensor::Int> {
    kv::selection_tensor::<TestBackend>(indices, &device()).expect("selector fits i32")
}

/// Builds one scalar-head attention layer with the selected storage and proposal capacity.
fn scalar_kv_spec(storage: KvCacheStorage, staging: usize) -> NativeKvCacheSpec {
    NativeKvCacheSpec::new(1, 1, 1, storage, DType::F32)
        .expect("valid KV spec")
        .with_staging_capacity(staging)
}

/// Builds one scalar recurrent layer with a two-token convolution history.
fn scalar_recurrent_spec(staging: usize) -> RecurrentStateSpec {
    RecurrentStateSpec::new(1, 1, 1, 1, 1, 2, DType::F32)
        .expect("valid recurrent spec")
        .with_staging_capacity(staging)
}

/// Rejects invalid tensor dimensions and dtypes and round-trips validated geometry.
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
    assert_eq!(recurrent.matrix_dtype(), DType::BF16);
    assert_eq!(recurrent.convolution_dtype(), DType::BF16);
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

/// Checks physical tensor extents for full, standard-ring, and mirrored storage.
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

/// Checks full-cache value preservation and rejects invalid indices and tensor metadata.
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
    assert!(
        cache
            .read_working_sequence(0, 0, 0)
            .expect("committed fallback")
            .materialize()
            .is_none()
    );
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

/// Compares wrapped standard and mirrored cache reads with chronological token values.
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
        assert_eq!(read.segments()[0].keys().dims()[0], 1);
        assert_eq!(read.segments()[0].values().dims()[0], 1);
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

/// Preserves committed ring tensors exactly when proposed writes are discarded.
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
    state.truncate(&[0]).expect("clear wrapped staged cache");
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

/// Rejects missing or excessive staging without changing committed values or the active mark.
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

/// Publishes the recurrent snapshot after the last accepted proposal token.
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

/// Reorders and duplicates attention, recurrent, and auxiliary state consistently.
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
    state.reset().expect("reset recurrent bank");
}

/// Clears all tensor storage and logical cursors through the architecture reset contract.
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

/// Allows recoverable rewinds and rejects missing ring or recurrent history.
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

/// Rejects incompatible batch geometry, lifecycle operations, and attached banks.
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

/// Locks bank topology and lifecycle mutations until the active transaction ends.
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

/// Handles empty inputs and arithmetic failures before invalid backend slicing.
#[test]
fn zero_length_and_arithmetic_edges_never_reach_invalid_backend_slices() {
    let device = device();
    let config = CacheConfig::new(1, 5).expect("config");
    let mut cache = NativeKvCache::<TestBackend>::new(
        config,
        1,
        scalar_kv_spec(KvCacheStorage::Full, 2),
        &device,
    )
    .expect("cache");
    let (keys, values) = scalar_kv(&[]);
    cache
        .write_sequence(0, 0, 5, keys, values)
        .expect("empty write at exclusive capacity");
    let (keys, values) = scalar_kv(&[1.0]);
    assert!(matches!(
        cache.write_sequence(0, 0, usize::MAX, keys, values),
        Err(TensorCacheError::WritePositionOverflow {
            start: usize::MAX,
            tokens: 1
        })
    ));

    cache.begin_staging().expect("staging");
    let (keys, values) = scalar_kv(&[]);
    cache
        .stage_sequence(0, 0, keys, values)
        .expect("empty proposal");
    assert_eq!(cache.staged_length(0, 0).expect("staged length"), 0);
    assert!(matches!(
        cache.staged_length(1, 0),
        Err(TensorCacheError::LayerOutOfBounds { .. })
    ));
    assert!(matches!(
        cache.staged_length(0, 1),
        Err(TensorCacheError::SequenceOutOfBounds { .. })
    ));
    cache.rollback_staging();

    let enormous = CacheConfig::new(1, usize::MAX)
        .expect("non-zero capacity")
        .with_sliding_window(usize::MAX)
        .expect("equal window");
    assert!(matches!(
        NativeKvCache::<TestBackend>::new(
            enormous,
            1,
            scalar_kv_spec(KvCacheStorage::MirroredSlidingRing, 0),
            &device
        ),
        Err(TensorCacheError::MirroredCapacityOverflow { window: usize::MAX })
    ));
    assert!(matches!(
        super::kv::selection_tensor::<TestBackend>(&[usize::MAX], &device),
        Err(TensorCacheError::BatchIndexOverflow { index: usize::MAX })
    ));
}

/// Rejects recurrent shape, row, layer, and transaction-lifecycle violations.
#[test]
fn recurrent_cache_validates_shapes_indices_and_staging_lifecycle() {
    let device = device();
    let config = CacheConfig::new(1, 5).expect("config");
    let spec = scalar_recurrent_spec(2);
    let mut cache =
        RecurrentStateCache::<TestBackend>::new(config, 1, spec, &device).expect("recurrent cache");
    assert_eq!(cache.config(), config);
    assert_eq!(cache.spec(), spec);
    assert_eq!(cache.device(), &device);
    assert!(matches!(
        cache.matrix(1),
        Err(TensorCacheError::LayerOutOfBounds { .. })
    ));
    assert!(matches!(
        cache.convolution(1),
        Err(TensorCacheError::LayerOutOfBounds { .. })
    ));

    let matrix = tensor4(vec![1.0], [1, 1, 1, 1]);
    let convolution = tensor3(vec![2.0, 3.0], [1, 1, 2]);
    cache
        .set_sequence_state(0, 0, matrix, convolution)
        .expect("valid state");
    assert!(matches!(
        cache.set_sequence_state(
            0,
            1,
            tensor4(vec![1.0], [1, 1, 1, 1]),
            tensor3(vec![1.0, 1.0], [1, 1, 2])
        ),
        Err(TensorCacheError::SequenceOutOfBounds { .. })
    ));
    assert!(matches!(
        cache.set_sequence_state(
            0,
            0,
            tensor4(vec![1.0, 2.0], [1, 1, 1, 2]),
            tensor3(vec![1.0, 1.0], [1, 1, 2])
        ),
        Err(TensorCacheError::ShapeMismatch {
            tensor: "recurrent matrix",
            ..
        })
    ));
    assert!(matches!(
        cache.set_sequence_state(
            0,
            0,
            tensor4(vec![1.0], [1, 1, 1, 1]),
            tensor3(vec![1.0], [1, 1, 1])
        ),
        Err(TensorCacheError::ShapeMismatch {
            tensor: "convolution state",
            ..
        })
    ));
    assert!(matches!(
        cache.set_sequence_state(
            0,
            0,
            Tensor::zeros([1, 1, 1, 1], (&device, DType::F64)),
            tensor3(vec![1.0, 1.0], [1, 1, 2])
        ),
        Err(TensorCacheError::DTypeMismatch {
            tensor: "recurrent matrix",
            ..
        })
    ));
    assert!(matches!(
        cache.stage_sequence_snapshots(
            0,
            0,
            tensor5(vec![1.0], [1, 1, 1, 1, 1]),
            tensor4(vec![1.0, 1.0], [1, 1, 1, 2])
        ),
        Err(TensorCacheError::TransactionNotActive)
    ));

    cache.begin_staging().expect("staging");
    assert!(cache.has_active_staging());
    assert!(matches!(
        cache.begin_staging(),
        Err(TensorCacheError::TransactionAlreadyActive)
    ));
    assert!(matches!(
        cache.set_sequence_state(
            0,
            0,
            tensor4(vec![1.0], [1, 1, 1, 1]),
            tensor3(vec![1.0, 1.0], [1, 1, 2])
        ),
        Err(TensorCacheError::TransactionActive { .. })
    ));
    cache
        .stage_sequence_snapshots(
            0,
            0,
            tensor5(vec![4.0], [1, 1, 1, 1, 1]),
            tensor4(vec![5.0, 6.0], [1, 1, 1, 2]),
        )
        .expect("one snapshot");
    assert_eq!(cache.staged_length(0, 0).expect("staged length"), 1);
    assert!(matches!(
        cache.staged_length(1, 0),
        Err(TensorCacheError::LayerOutOfBounds { .. })
    ));
    assert!(matches!(
        cache.staged_length(0, 1),
        Err(TensorCacheError::SequenceOutOfBounds { .. })
    ));
    assert!(matches!(
        cache.stage_sequence_snapshots(
            0,
            0,
            tensor5(vec![7.0, 8.0], [1, 2, 1, 1, 1]),
            tensor4(vec![7.0, 7.0, 8.0, 8.0], [1, 2, 1, 2])
        ),
        Err(TensorCacheError::StagingCapacityExceeded {
            requested: 3,
            capacity: 2
        })
    ));
    assert!(matches!(
        cache.prepare_staging_commit(&[2], &[1]),
        Err(TensorCacheError::IncompleteStaging {
            expected: 2,
            actual: 1,
            ..
        })
    ));
    cache.rollback_staging();
    assert!(!cache.has_active_staging());
    assert!(matches!(
        cache.prepare_staging_commit(&[0], &[0]),
        Err(TensorCacheError::TransactionNotActive)
    ));
    cache.reorder_rows(&selection(&[0]));
    cache.reset();

    let mut no_staging =
        RecurrentStateCache::<TestBackend>::new(config, 1, scalar_recurrent_spec(0), &device)
            .expect("cache");
    assert!(matches!(
        no_staging.begin_staging(),
        Err(TensorCacheError::StagingDisabled)
    ));
}

/// Keeps auxiliary accepted progress and tensor storage independent of the main decoder.
#[test]
fn mtp_transactions_commit_and_rollback_without_aliasing_main_state() {
    let device = device();
    let config = CacheConfig::new(1, 6).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 2))
        .expect("main cache");
    state
        .enable_mtp()
        .expect("MTP")
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 2))
        .expect("MTP cache");
    let (keys, values) = scalar_kv(&[1.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .write_sequence(0, 0, 0, keys, values)
        .expect("main baseline");
    let (keys, values) = scalar_kv(&[10.0]);
    state
        .mtp_mut()
        .expect("MTP")
        .native_kv_mut(0)
        .expect("cache")
        .write_sequence(0, 0, 0, keys, values)
        .expect("MTP baseline");
    state.advance(&[1]).expect("baseline");
    state.advance_mtp(&[1]).expect("MTP baseline");

    let mark = state.begin_transaction().expect("transaction");
    let (keys, values) = scalar_kv(&[2.0, 3.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .stage_sequence(0, 0, keys, values)
        .expect("main proposal");
    let (keys, values) = scalar_kv(&[20.0, 30.0]);
    state
        .mtp_mut()
        .expect("MTP")
        .native_kv_mut(0)
        .expect("cache")
        .stage_sequence(0, 0, keys, values)
        .expect("MTP proposal");
    state.advance(&[2]).expect("proposal");
    state.advance_mtp(&[2]).expect("MTP proposal");
    state
        .commit_uniform_prefixes(mark, 1, Some(1))
        .expect("uniform prefix");
    assert_eq!(state.lengths(), &[2]);
    assert_eq!(state.mtp_lengths(), Some([2].as_slice()));
    assert_eq!(
        materialized_keys(state.main().native_kv(0).expect("cache"), 0, 0, 2),
        [1.0, 2.0]
    );
    assert_eq!(
        materialized_keys(
            state.mtp().expect("MTP").native_kv(0).expect("cache"),
            0,
            0,
            2
        ),
        [10.0, 20.0]
    );

    let mark = state.begin_transaction().expect("transaction");
    for bank in [false, true] {
        let (keys, values) = scalar_kv(&[if bank { 40.0 } else { 4.0 }]);
        let target = if bank {
            state.mtp_mut().expect("MTP")
        } else {
            state.main_mut()
        };
        target
            .native_kv_mut(0)
            .expect("cache")
            .stage_sequence(0, 0, keys, values)
            .expect("proposal");
    }
    state.advance(&[1]).expect("proposal");
    state.advance_mtp(&[1]).expect("MTP proposal");
    state.rollback(mark).expect("rollback");
    assert_eq!(
        materialized_keys(state.main().native_kv(0).expect("cache"), 0, 0, 2),
        [1.0, 2.0]
    );
    state.reset().expect("reset main and MTP");
    assert_eq!(state.lengths(), &[0]);
    assert_eq!(state.mtp_lengths(), Some([0].as_slice()));
    assert!(
        state
            .mtp()
            .expect("MTP")
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

/// Rejects foreign marks and excessive acceptance while allowing valid recovery.
#[test]
fn transaction_marks_and_accepted_counts_are_state_local_and_recoverable() {
    let device = device();
    let config = CacheConfig::new(2, 5).expect("config");
    let mut first = TensorDecodeState::<TestBackend>::new(config, 2, &device).expect("state");
    let mut second = TensorDecodeState::<TestBackend>::new(config, 2, &device).expect("state");
    let first_mark = first.begin_transaction().expect("transaction");
    assert!(format!("{first_mark:?}").contains("TransactionToken(..)"));
    let rollback = first_mark.clone();
    let foreign = second.begin_transaction().expect("transaction");
    let second_rollback = foreign.clone();
    assert!(matches!(
        first.commit_prefix(foreign, &[0, 0]),
        Err(TensorCacheError::StaleTransactionMark)
    ));
    assert!(matches!(
        first.commit_prefix(first_mark, &[0]),
        Err(TensorCacheError::Metadata(
            crate::cache::DecodeStateError::SequenceCountMismatch { .. }
        ))
    ));
    assert!(first.has_active_transaction());
    first.advance(&[1, 0]).expect("proposal");
    assert!(matches!(
        first.commit_prefix(rollback.clone(), &[2, 0]),
        Err(TensorCacheError::AcceptedPrefixExceedsProposal {
            sequence: 0,
            accepted: 2,
            proposed: 1
        })
    ));
    first.rollback(rollback).expect("rollback after errors");
    second.rollback(second_rollback).expect("second rollback");
}

/// Validates bank compatibility and reports unavailable group indices with context.
#[test]
fn bank_attachment_accessors_and_errors_are_contextual() {
    let device = device();
    let config = CacheConfig::new(1, 3).expect("config");
    assert!(matches!(
        TensorCacheBank::<TestBackend>::new(config, 2, &device),
        Err(TensorCacheError::BatchSizeMismatch { .. })
    ));
    let mut bank = TensorCacheBank::<TestBackend>::new(config, 1, &device).expect("bank");
    assert_eq!(bank.config(), config);
    assert_eq!(bank.batch_size(), 1);
    assert_eq!(bank.native_kv_count(), 0);
    assert_eq!(bank.recurrent_count(), 0);
    bank.add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 0))
        .expect("KV");
    bank.add_recurrent(scalar_recurrent_spec(0))
        .expect("recurrent");
    assert_eq!(bank.native_kv_count(), 1);
    assert_eq!(bank.recurrent_count(), 1);
    assert!(matches!(
        bank.native_kv_mut(1),
        Err(TensorCacheError::CacheOutOfBounds { .. })
    ));
    assert!(matches!(
        bank.recurrent_mut(1),
        Err(TensorCacheError::CacheOutOfBounds { .. })
    ));

    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state.attach_mtp(bank).expect("compatible bank");
    assert!(state.mtp().is_some());
    assert_eq!(state.config(), config);
    assert_eq!(state.batch_size(), 1);
    assert_eq!(state.capacity(), 3);

    let runtime: crate::Error = TensorCacheError::StagingDisabled.into();
    assert_eq!(
        runtime.to_string(),
        "tensor cache error: speculative staging is disabled; configure a non-zero staging capacity"
    );
    assert!(matches!(
        runtime,
        crate::Error::TensorCache(TensorCacheError::StagingDisabled)
    ));
    assert!(
        TensorCacheError::ShapeMismatch {
            tensor: "keys",
            expected: vec![1, 2],
            actual: vec![2, 1],
        }
        .to_string()
        .contains("expected [1, 2]")
    );
}

/// Checks direct staging lifecycle and zero or positive accepted prefixes across storage modes.
#[test]
fn low_level_commit_validation_covers_full_mirrored_and_zero_prefixes() {
    let device = device();
    let full_config = CacheConfig::new(1, 4).expect("config");
    let mut full = NativeKvCache::<TestBackend>::new(
        full_config,
        1,
        scalar_kv_spec(KvCacheStorage::Full, 2),
        &device,
    )
    .expect("full cache");
    let (keys, values) = scalar_kv(&[1.0]);
    assert!(matches!(
        full.stage_sequence(0, 0, keys, values),
        Err(TensorCacheError::TransactionNotActive)
    ));
    full.begin_staging().expect("staging");
    assert!(matches!(
        full.begin_staging(),
        Err(TensorCacheError::TransactionAlreadyActive)
    ));
    let (keys, values) = scalar_kv(&[1.0]);
    full.stage_sequence(0, 0, keys, values).expect("proposal");
    assert!(matches!(
        full.prepare_staging_commit(&[], &[1], &[1]),
        Err(TensorCacheError::Metadata(
            crate::cache::DecodeStateError::SequenceCountMismatch { .. }
        ))
    ));
    assert!(matches!(
        full.prepare_staging_commit(&[0], &[1], &[2]),
        Err(TensorCacheError::AcceptedPrefixExceedsProposal { .. })
    ));
    full.rollback_staging();

    full.begin_staging().expect("empty proposal");
    let plan = full
        .prepare_staging_commit(&[0], &[0], &[0])
        .expect("zero proposal valid");
    full.apply_staging_commit(plan);
    assert!(matches!(
        full.prepare_staging_commit(&[0], &[0], &[0]),
        Err(TensorCacheError::TransactionNotActive)
    ));
    full.reorder_rows(&selection(&[0]));

    let mut disabled = NativeKvCache::<TestBackend>::new(
        full_config,
        1,
        scalar_kv_spec(KvCacheStorage::Full, 0),
        &device,
    )
    .expect("cache");
    assert!(matches!(
        disabled.begin_staging(),
        Err(TensorCacheError::StagingDisabled)
    ));

    let mirrored_config = CacheConfig::new(1, 8)
        .expect("config")
        .with_sliding_window(3)
        .expect("window");
    let mut state =
        TensorDecodeState::<TestBackend>::new(mirrored_config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::MirroredSlidingRing, 2))
        .expect("mirrored cache");
    let mark = state.begin_transaction().expect("transaction");
    let (keys, values) = scalar_kv(&[6.0, 7.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .stage_sequence(0, 0, keys, values)
        .expect("proposal");
    state.advance(&[2]).expect("proposal length");
    state.commit_prefix(mark, &[2]).expect("commit mirrored");
    assert_eq!(
        materialized_keys(state.main().native_kv(0).expect("cache"), 0, 0, 2),
        [6.0, 7.0]
    );
}

/// Preserves recurrent terminal values for zero acceptance and empty snapshots.
#[test]
fn recurrent_zero_prefix_and_empty_snapshots_keep_terminal_state_exact() {
    let device = device();
    let config = CacheConfig::new(1, 4).expect("config");
    let mut cache =
        RecurrentStateCache::<TestBackend>::new(config, 1, scalar_recurrent_spec(2), &device)
            .expect("cache");
    cache.begin_staging().expect("transaction");
    let plan = cache
        .prepare_staging_commit(&[0], &[0])
        .expect("empty staging");
    cache.apply_staging_commit(plan);
    assert_eq!(
        cache
            .matrix(0)
            .expect("matrix")
            .to_data()
            .to_vec::<f32>()
            .expect("f32"),
        [0.0]
    );

    cache.begin_staging().expect("transaction");
    cache
        .stage_sequence_snapshots(
            0,
            0,
            tensor5(vec![1.0], [1, 1, 1, 1, 1]),
            tensor4(vec![1.0, 2.0], [1, 1, 1, 2]),
        )
        .expect("snapshot");
    assert!(matches!(
        cache.prepare_staging_commit(&[1], &[2]),
        Err(TensorCacheError::AcceptedPrefixExceedsProposal { .. })
    ));
    cache.rollback_staging();

    cache.begin_staging().expect("transaction");
    cache
        .stage_sequence_snapshots(
            0,
            0,
            tensor5(Vec::new(), [1, 0, 1, 1, 1]),
            tensor4(Vec::new(), [1, 0, 1, 2]),
        )
        .expect("empty snapshots");
    cache.rollback_staging();
}

/// Applies independent auxiliary rewinds and architecture lifecycle operations safely.
#[test]
fn mtp_truncate_and_trait_lifecycle_apply_to_auxiliary_storage() {
    let device = device();
    let config = CacheConfig::new(1, 6)
        .expect("config")
        .with_sliding_window(2)
        .expect("window");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .enable_mtp()
        .expect("MTP")
        .add_native_kv(scalar_kv_spec(KvCacheStorage::SlidingRing, 0))
        .expect("MTP ring");
    state.advance_mtp(&[3]).expect("wrapped MTP length");
    assert!(matches!(
        state.truncate_mtp(&[2]),
        Err(TensorCacheError::UnsafeWrappedTruncate { .. })
    ));
    state.truncate_mtp(&[0]).expect("zero MTP rewind");
    ArchitectureDecodeState::<TestBackend>::truncate(&mut state, &[0])
        .expect("main cursor is independent");
    ArchitectureDecodeState::<TestBackend>::reorder(&mut state, &[0]).expect("trait reorder");

    let bank = TensorCacheBank::<TestBackend>::new(config, 1, &device).expect("bank");
    assert!(matches!(
        state.attach_mtp(bank),
        Err(TensorCacheError::MtpAlreadyEnabled)
    ));
}

/// Preserves nonzero main and auxiliary tensors when either requested rewind is unsafe.
#[test]
fn paired_main_mtp_truncate_validates_both_banks_before_mutation() {
    let device = device();
    let config = CacheConfig::new(1, 6)
        .expect("config")
        .with_sliding_window(2)
        .expect("window");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 1))
        .expect("main cache");
    state
        .enable_mtp()
        .expect("MTP")
        .add_native_kv(scalar_kv_spec(KvCacheStorage::SlidingRing, 1))
        .expect("MTP ring");
    state.advance(&[3]).expect("main length");
    state.advance_mtp(&[3]).expect("MTP length");

    let (keys, values) = scalar_kv(&[1.0, 2.0, 3.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("main cache")
        .write_sequence(0, 0, 0, keys, values)
        .expect("main sentinel values");
    let (keys, values) = scalar_kv(&[7.0, 8.0, 9.0]);
    state
        .mtp_mut()
        .expect("MTP bank")
        .native_kv_mut(0)
        .expect("MTP cache")
        .write_sequence(0, 0, 0, keys, values)
        .expect("MTP sentinel values");

    assert!(matches!(
        state.truncate_prefixes(&[0], &[2]),
        Err(TensorCacheError::UnsafeWrappedTruncate { .. })
    ));
    assert_eq!(state.lengths(), &[3]);
    assert_eq!(state.mtp_lengths(), Some([3].as_slice()));
    assert_eq!(
        materialized_keys(state.main().native_kv(0).expect("main cache"), 0, 0, 3),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(
        materialized_keys(
            state
                .mtp()
                .expect("MTP bank")
                .native_kv(0)
                .expect("MTP cache"),
            0,
            0,
            3
        ),
        [8.0, 9.0]
    );

    let mark = state.begin_transaction().expect("transaction");
    assert!(matches!(
        state.truncate_prefixes(&[0], &[0]),
        Err(TensorCacheError::TransactionActive { .. })
    ));
    state.rollback(mark).expect("rollback");

    state
        .truncate_prefixes(&[0], &[0])
        .expect("paired zero rewind");
    assert_eq!(state.lengths(), &[0]);
    assert_eq!(state.mtp_lengths(), Some([0].as_slice()));
}

/// A mirrored commit crossing the ring boundary retains chronology and duplicates both halves.
#[test]
fn mirrored_commit_crosses_wrap_with_independent_layer_values() {
    let config = CacheConfig::new(1, 8)
        .expect("config")
        .with_sliding_window(3)
        .expect("window");
    let spec = NativeKvCacheSpec::new(2, 1, 1, KvCacheStorage::MirroredSlidingRing, DType::F32)
        .expect("geometry")
        .with_staging_capacity(3);
    let mut cache = NativeKvCache::<TestBackend>::new(config, 1, spec, &device()).expect("cache");
    for layer in 0..2 {
        let offset = (layer * 10) as f32;
        let (keys, values) = scalar_kv(&[offset + 1.0, offset + 2.0]);
        cache
            .write_sequence(layer, 0, 0, keys, values)
            .expect("baseline");
    }
    cache.begin_staging().expect("staging");
    for layer in 0..2 {
        let offset = (layer * 10) as f32;
        let (keys, values) = scalar_kv(&[offset + 3.0, offset + 4.0, offset + 5.0]);
        cache
            .stage_sequence(layer, 0, keys, values)
            .expect("proposal");
    }
    let plan = cache
        .prepare_staging_commit(&[2], &[3], &[2])
        .expect("accepted prefix");
    cache.apply_staging_commit(plan);
    for layer in 0..2 {
        let offset = (layer * 10) as f32;
        assert_eq!(
            materialized_keys(&cache, layer, 0, 4),
            [offset + 2.0, offset + 3.0, offset + 4.0]
        );
        for tensor in [
            cache.keys(layer).expect("keys"),
            cache.values(layer).expect("values"),
        ] {
            let data = tensor.clone().into_data().to_vec::<f32>().expect("F32");
            assert_eq!(data[..3], data[3..]);
        }
    }
}

/// Commits different accepted prefix lengths for rows sharing one tensor cache.
#[test]
fn ragged_batch_transactions_commit_independent_prefix_lengths() {
    let device = device();
    let config = CacheConfig::new(2, 6).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 2, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 3))
        .expect("cache");
    let mark = state.begin_transaction().expect("transaction");
    for (sequence, proposal) in [(0, vec![1.0, 2.0]), (1, vec![10.0])] {
        let (keys, values) = scalar_kv(&proposal);
        state
            .main_mut()
            .native_kv_mut(0)
            .expect("cache")
            .stage_sequence(0, sequence, keys, values)
            .expect("ragged proposal");
    }
    state.advance(&[2, 1]).expect("ragged lengths");
    state.commit_prefix(mark, &[1, 0]).expect("ragged commit");
    assert_eq!(state.lengths(), &[1, 0]);
    assert_eq!(
        materialized_keys(state.main().native_kv(0).expect("cache"), 0, 0, 1),
        [1.0]
    );
    assert!(
        state
            .main()
            .native_kv(0)
            .expect("cache")
            .read_sequence(0, 1, 0)
            .expect("empty row")
            .materialize()
            .is_none()
    );
}

/// Keeps committed cursors fixed while working reads reveal each newly staged token.
#[test]
fn committed_lengths_stay_stable_while_working_kv_grows_iteratively() {
    let device = device();
    let config = CacheConfig::new(1, 8).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 3))
        .expect("cache");
    let (keys, values) = scalar_kv(&[1.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .write_sequence(0, 0, 0, keys, values)
        .expect("baseline");
    state.advance(&[1]).expect("committed baseline");

    let mark = state.begin_transaction().expect("transaction");
    for value in [2.0, 3.0] {
        let (keys, values) = scalar_kv(&[value]);
        state
            .main_mut()
            .native_kv_mut(0)
            .expect("cache")
            .stage_sequence(0, 0, keys, values)
            .expect("stage iterative token");
        state.advance(&[1]).expect("working advance");
        assert_eq!(state.lengths(), &[1]);
        let expected = if value == 2.0 {
            vec![1.0, 2.0]
        } else {
            vec![1.0, 2.0, 3.0]
        };
        assert_eq!(state.working_lengths(), &[expected.len()]);
        assert_eq!(
            working_keys(state.main().native_kv(0).expect("cache"), 0, 0, 1),
            expected
        );
    }
    state.rollback(mark).expect("rollback");
    assert_eq!(state.lengths(), &[1]);
    assert_eq!(state.working_lengths(), &[1]);
}

/// Trims committed history and oversized proposals to the correct visible ring suffix.
#[test]
fn working_reads_trim_wrapped_history_and_support_proposals_larger_than_window() {
    let device = device();
    let config = CacheConfig::new(1, 12)
        .expect("config")
        .with_sliding_window(4)
        .expect("window");
    for storage in [
        KvCacheStorage::SlidingRing,
        KvCacheStorage::MirroredSlidingRing,
    ] {
        let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
        state
            .main_mut()
            .add_native_kv(scalar_kv_spec(storage, 5))
            .expect("cache");
        let (keys, values) = scalar_kv(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        state
            .main_mut()
            .native_kv_mut(0)
            .expect("cache")
            .write_sequence(0, 0, 0, keys, values)
            .expect("wrapped baseline");
        state.advance(&[5]).expect("baseline length");

        let mark = state.begin_transaction().expect("transaction");
        let (keys, values) = scalar_kv(&[6.0, 7.0]);
        state
            .main_mut()
            .native_kv_mut(0)
            .expect("cache")
            .stage_sequence(0, 0, keys, values)
            .expect("proposal");
        state.advance(&[2]).expect("proposal length");
        let read = state
            .main()
            .native_kv(0)
            .expect("cache")
            .read_working_sequence(0, 0, 5)
            .expect("working read");
        assert_eq!(read.retained_positions(), 3..7);
        assert_eq!(
            working_keys(state.main().native_kv(0).expect("cache"), 0, 0, 5),
            [4.0, 5.0, 6.0, 7.0]
        );
        state.rollback(mark).expect("rollback");

        let mark = state.begin_transaction().expect("transaction");
        let (keys, values) = scalar_kv(&[6.0, 7.0, 8.0, 9.0, 10.0]);
        state
            .main_mut()
            .native_kv_mut(0)
            .expect("cache")
            .stage_sequence(0, 0, keys, values)
            .expect("large proposal");
        state.advance(&[5]).expect("proposal length");
        assert_eq!(
            working_keys(state.main().native_kv(0).expect("cache"), 0, 0, 5),
            [7.0, 8.0, 9.0, 10.0]
        );
        state.rollback(mark).expect("rollback");
    }
}

/// Preserves independent recurrent dtypes through working reads and batched state updates.
#[test]
fn recurrent_working_reads_and_whole_batch_io_preserve_split_dtypes() {
    let device = device();
    let config = CacheConfig::new(2, 8).expect("config");
    let spec = RecurrentStateSpec::new_with_dtypes(1, 1, 1, 1, 1, 2, DType::F32, DType::F16)
        .expect("mixed dtype spec")
        .with_staging_capacity(2);
    assert_eq!(spec.matrix_dtype(), DType::F32);
    assert_eq!(spec.convolution_dtype(), DType::F16);
    let mut cache =
        RecurrentStateCache::<TestBackend>::new(config, 2, spec, &device).expect("cache");
    let initial = cache.read_working_batch(0).expect("initial batch");
    assert_eq!(initial.staged_rows(), &[false, false]);
    assert!(matches!(
        cache.stage_batch_snapshots(
            0,
            tensor5(vec![0.0, 0.0], [2, 1, 1, 1, 1]),
            tensor4(vec![0.0; 4], [2, 1, 1, 2]).cast(DType::F16)
        ),
        Err(TensorCacheError::TransactionNotActive)
    ));
    assert!(matches!(
        cache.set_batch_state(
            0,
            tensor4(vec![0.0], [1, 1, 1, 1]),
            tensor3(vec![0.0; 4], [2, 1, 2]).cast(DType::F16)
        ),
        Err(TensorCacheError::ShapeMismatch { .. })
    ));
    let matrix = tensor4(vec![1.0, 2.0], [2, 1, 1, 1]);
    let convolution = tensor3(vec![3.0, 4.0, 5.0, 6.0], [2, 1, 2]).cast(DType::F16);
    cache
        .set_batch_state(0, matrix, convolution)
        .expect("whole-batch set");
    let (matrix, convolution) = cache.batch_state(0).expect("whole-batch get");
    assert_eq!(matrix.dims(), [2, 1, 1, 1]);
    assert_eq!(matrix.dtype(), DType::F32);
    assert_eq!(convolution.dims(), [2, 1, 2]);
    assert_eq!(convolution.dtype(), DType::F16);
    let committed = cache.read_working_sequence(0, 1).expect("committed row");
    assert_eq!(committed.staged_snapshot(), None);
    assert_eq!(committed.matrix().dims(), [1, 1, 1, 1]);

    cache.begin_staging().expect("uniform batch transaction");
    cache
        .stage_batch_snapshots(
            0,
            tensor5(Vec::new(), [2, 0, 1, 1, 1]),
            tensor4(Vec::new(), [2, 0, 1, 2]).cast(DType::F16),
        )
        .expect("empty batch snapshots");
    assert!(matches!(
        cache.set_batch_state(
            0,
            tensor4(vec![1.0, 2.0], [2, 1, 1, 1]),
            tensor3(vec![3.0, 4.0, 5.0, 6.0], [2, 1, 2]).cast(DType::F16)
        ),
        Err(TensorCacheError::TransactionActive { .. })
    ));
    cache
        .stage_batch_snapshots(
            0,
            tensor5(vec![7.0, 8.0], [2, 1, 1, 1, 1]),
            tensor4(vec![9.0, 10.0, 11.0, 12.0], [2, 1, 1, 2]).cast(DType::F16),
        )
        .expect("whole-batch snapshots");
    assert!(matches!(
        cache.stage_batch_snapshots(
            0,
            tensor5(vec![1.0, 2.0, 3.0, 4.0], [2, 2, 1, 1, 1]),
            tensor4(vec![1.0; 8], [2, 2, 1, 2]).cast(DType::F16)
        ),
        Err(TensorCacheError::StagingCapacityExceeded {
            requested: 3,
            capacity: 2
        })
    ));
    let uniform = cache.read_working_batch(0).expect("uniform working batch");
    assert_eq!(uniform.staged_rows(), &[true, true]);
    assert_eq!(
        uniform
            .matrix()
            .to_data()
            .to_vec::<f32>()
            .expect("matrix data"),
        [7.0, 8.0]
    );
    cache.rollback_staging();

    cache.begin_staging().expect("ragged transaction");
    cache
        .stage_sequence_snapshots(
            0,
            1,
            tensor5(vec![7.0, 8.0], [1, 2, 1, 1, 1]),
            tensor4(vec![9.0, 10.0, 11.0, 12.0], [1, 2, 1, 2]).cast(DType::F16),
        )
        .expect("snapshots");
    assert!(matches!(
        cache.stage_batch_snapshots(
            0,
            tensor5(vec![1.0, 2.0], [2, 1, 1, 1, 1]),
            tensor4(vec![1.0, 1.0, 2.0, 2.0], [2, 1, 1, 2]).cast(DType::F16)
        ),
        Err(TensorCacheError::NonUniformStagingOffset { .. })
    ));
    let working = cache.read_working_sequence(0, 1).expect("working row");
    assert_eq!(working.staged_snapshot(), Some(1));
    assert_eq!(
        working
            .matrix()
            .to_data()
            .to_vec::<f32>()
            .expect("matrix data"),
        [8.0]
    );
    assert_eq!(working.convolution().dtype(), DType::F16);
    let working_batch = cache.read_working_batch(0).expect("working batch");
    assert_eq!(working_batch.staged_rows(), &[false, true]);
    assert_eq!(working_batch.matrix().dims(), [2, 1, 1, 1]);
    assert_eq!(
        working_batch
            .matrix()
            .to_data()
            .to_vec::<f32>()
            .expect("matrix data"),
        [1.0, 8.0]
    );
    assert_eq!(working_batch.convolution().dtype(), DType::F16);
    cache.rollback_staging();

    assert!(matches!(
        RecurrentStateSpec::new_with_dtypes(1, 1, 1, 1, 1, 1, DType::F32, DType::I32),
        Err(TensorCacheError::NonFloatDType { dtype: DType::I32 })
    ));
}

/// Tracks separate main and auxiliary proposal cursors and accepted prefix lengths.
#[test]
fn main_and_mtp_proposals_have_independent_cursors_and_prefixes() {
    let device = device();
    let config = CacheConfig::new(1, 8).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 3))
        .expect("main cache");
    state
        .enable_mtp()
        .expect("MTP")
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 3))
        .expect("MTP cache");
    let mark = state.begin_transaction().expect("transaction");
    assert_eq!(mark.mtp_baseline_lengths(), Some([0].as_slice()));
    let (keys, values) = scalar_kv(&[1.0, 2.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("main")
        .stage_sequence(0, 0, keys, values)
        .expect("main proposal");
    let (keys, values) = scalar_kv(&[10.0]);
    state
        .mtp_mut()
        .expect("MTP")
        .native_kv_mut(0)
        .expect("cache")
        .stage_sequence(0, 0, keys, values)
        .expect("MTP proposal");
    state.advance(&[2]).expect("main cursor");
    state.advance_mtp(&[1]).expect("MTP cursor");
    assert_eq!(state.lengths(), &[0]);
    assert_eq!(state.working_lengths(), &[2]);
    assert_eq!(state.mtp_lengths(), Some([0].as_slice()));
    assert_eq!(state.mtp_working_lengths(), Some([1].as_slice()));
    state
        .commit_prefixes(mark, &[1], Some(&[0]))
        .expect("independent commit");
    assert_eq!(state.lengths(), &[1]);
    assert_eq!(state.mtp_lengths(), Some([0].as_slice()));
    assert_eq!(
        materialized_keys(state.main().native_kv(0).expect("main"), 0, 0, 1),
        [1.0]
    );

    let mark = state.begin_transaction().expect("transaction");
    let (keys, values) = scalar_kv(&[20.0]);
    state
        .mtp_mut()
        .expect("MTP")
        .native_kv_mut(0)
        .expect("cache")
        .stage_sequence(0, 0, keys, values)
        .expect("MTP proposal");
    state.advance(&[0]).expect("empty main proposal");
    state.advance_mtp(&[1]).expect("MTP proposal");
    assert!(matches!(
        state.commit_prefix(mark.clone(), &[0]),
        Err(TensorCacheError::MtpAcceptedPrefixRequired)
    ));
    assert!(state.has_active_transaction());
    state.rollback(mark).expect("recoverable rollback");
}

/// Exercises tensor transaction semantics through the public generic lifecycle trait.
#[test]
fn public_transaction_trait_drives_tensor_state_without_downcasting() {
    use crate::cache::TransactionalCache;

    let device = device();
    let config = CacheConfig::new(1, 4).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 1))
        .expect("cache");
    let mark = TransactionalCache::mark(&mut state).expect("generic mark");
    let (keys, values) = scalar_kv(&[4.0]);
    state
        .main_mut()
        .native_kv_mut(0)
        .expect("cache")
        .stage_sequence(0, 0, keys, values)
        .expect("proposal");
    state.advance(&[1]).expect("working advance");
    TransactionalCache::commit_prefix(&mut state, mark, 1).expect("generic commit");
    assert_eq!(state.lengths(), &[1]);
}

/// Locks initially empty banks against adding groups during speculation.
#[test]
fn empty_banks_are_topology_locked_for_the_entire_transaction() {
    let device = device();
    let config = CacheConfig::new(1, 4).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state.enable_mtp().expect("empty MTP");
    let mark = state.begin_transaction().expect("empty transaction");
    assert!(state.main().has_active_staging());
    assert!(state.mtp().expect("MTP").has_active_staging());
    assert!(matches!(
        state
            .main_mut()
            .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 1)),
        Err(TensorCacheError::TransactionActive { .. })
    ));
    assert!(matches!(
        state
            .mtp_mut()
            .expect("MTP")
            .add_recurrent(scalar_recurrent_spec(1)),
        Err(TensorCacheError::TransactionActive { .. })
    ));
    state.rollback(mark).expect("rollback");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 1))
        .expect("topology unlocked");
}

/// Rejects invalid auxiliary staging before publishing any main-bank tensor.
#[test]
fn cross_bank_commit_validation_is_atomic_before_any_tensor_write() {
    let device = device();
    let config = CacheConfig::new(1, 8).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    state
        .main_mut()
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 2))
        .expect("main cache");
    state
        .enable_mtp()
        .expect("MTP")
        .add_native_kv(scalar_kv_spec(KvCacheStorage::Full, 2))
        .expect("MTP cache");
    for (mtp, value) in [(false, 1.0), (true, 10.0)] {
        let (keys, values) = scalar_kv(&[value]);
        let bank = if mtp {
            state.mtp_mut().expect("MTP")
        } else {
            state.main_mut()
        };
        bank.native_kv_mut(0)
            .expect("cache")
            .write_sequence(0, 0, 0, keys, values)
            .expect("baseline");
    }
    state.advance(&[1]).expect("main baseline");
    state.advance_mtp(&[1]).expect("MTP baseline");
    let mark = state.begin_transaction().expect("transaction");
    for (mtp, value) in [(false, 2.0), (true, 20.0)] {
        let (keys, values) = scalar_kv(&[value]);
        let bank = if mtp {
            state.mtp_mut().expect("MTP")
        } else {
            state.main_mut()
        };
        bank.native_kv_mut(0)
            .expect("cache")
            .stage_sequence(0, 0, keys, values)
            .expect("proposal");
    }
    state.advance(&[1]).expect("main proposal");
    state
        .advance_mtp(&[2])
        .expect("intentionally incomplete MTP proposal");
    assert!(matches!(
        state.commit_prefixes(mark.clone(), &[1], Some(&[1])),
        Err(TensorCacheError::IncompleteStaging {
            kind: "native KV",
            expected: 2,
            actual: 1,
            ..
        })
    ));
    assert_eq!(
        materialized_keys(state.main().native_kv(0).expect("main"), 0, 0, 1),
        [1.0]
    );
    assert_eq!(
        materialized_keys(
            state.mtp().expect("MTP").native_kv(0).expect("cache"),
            0,
            0,
            1
        ),
        [10.0]
    );
    state
        .rollback(mark)
        .expect("rollback after validation error");
}

/// Rejects invalid allocation geometry before invoking the tensor backend.
#[test]
fn allocation_arithmetic_is_rejected_before_backend_tensor_creation() {
    let device = device();
    let overflow_config = CacheConfig::new(1, usize::MAX).expect("config");
    let overflow_spec =
        NativeKvCacheSpec::new(1, 2, 2, KvCacheStorage::Full, DType::F32).expect("spec");
    assert!(matches!(
        NativeKvCache::<TestBackend>::new(overflow_config, 1, overflow_spec, &device),
        Err(TensorCacheError::TensorShapeOverflow { .. })
    ));

    let too_large = (i32::MAX as usize) + 1;
    let limit_config = CacheConfig::new(1, too_large).expect("config");
    let limit_spec = scalar_kv_spec(KvCacheStorage::Full, 0);
    assert!(matches!(
        NativeKvCache::<TestBackend>::new(limit_config, 1, limit_spec, &device),
        Err(TensorCacheError::TensorElementLimitExceeded {
            elements,
            limit,
            ..
        }) if elements == too_large && limit == i32::MAX as usize
    ));

    let repeated_spec =
        NativeKvCacheSpec::new(usize::MAX, 1, 1, KvCacheStorage::Full, DType::F32).expect("spec");
    let tiny_config = CacheConfig::new(1, 1).expect("config");
    assert!(matches!(
        NativeKvCache::<TestBackend>::new(tiny_config, 1, repeated_spec, &device),
        Err(TensorCacheError::AllocationSizeOverflow { .. })
            | Err(TensorCacheError::TensorShapeOverflow { .. })
    ));

    let recurrent_spec =
        RecurrentStateSpec::new(1, 1, usize::MAX, 2, 1, 1, DType::F32).expect("spec");
    assert!(matches!(
        RecurrentStateCache::<TestBackend>::new(tiny_config, 1, recurrent_spec, &device),
        Err(TensorCacheError::TensorShapeOverflow { .. })
    ));
}

/// Checks invalid transaction and auxiliary-prefix requests preserve recoverability.
#[test]
fn tensor_transaction_edge_contracts_remain_recoverable() {
    use crate::cache::TransactionalCache;

    let device = device();
    let config = CacheConfig::new(1, 2).expect("config");
    let mut state = TensorDecodeState::<TestBackend>::new(config, 1, &device).expect("state");
    assert_eq!(state.mtp_lengths(), None);
    assert_eq!(state.mtp_working_lengths(), None);
    assert!(matches!(
        state.advance_mtp(&[1]),
        Err(TensorCacheError::MtpNotEnabled)
    ));
    assert!(matches!(
        state.truncate_mtp(&[0]),
        Err(TensorCacheError::MtpNotEnabled)
    ));

    let mark = TransactionalCache::mark(&mut state).expect("generic mark");
    assert!(matches!(
        state.advance(&[3]),
        Err(TensorCacheError::Metadata(
            crate::cache::DecodeStateError::CapacityExceeded { .. }
        ))
    ));
    assert_eq!(state.working_lengths(), &[0]);
    TransactionalCache::rollback(&mut state, mark).expect("generic rollback");

    let mark = state.begin_transaction().expect("transaction");
    assert!(matches!(
        state.commit_prefixes(mark.clone(), &[0], Some(&[0])),
        Err(TensorCacheError::MtpNotEnabled)
    ));
    state.rollback(mark).expect("rollback");

    state.enable_mtp().expect("empty MTP");
    let mark = state.begin_transaction().expect("transaction");
    state
        .commit_prefix(mark, &[0])
        .expect("implicit zero MTP prefix");
    state.advance_mtp(&[1]).expect("MTP length");
    assert!(matches!(
        state.truncate_mtp(&[2]),
        Err(TensorCacheError::Metadata(
            crate::cache::DecodeStateError::CannotExtendByTruncation { .. }
        ))
    ));
}

/// Invalid snapshots cannot overwrite either an earlier proposal or committed state.
fn assert_snapshot_contract_rejections_are_atomic(whole_batch: bool) {
    let config = CacheConfig::new(2, 5).expect("small two-row cache");
    let spec = RecurrentStateSpec::new_with_dtypes(1, 1, 1, 1, 1, 2, DType::F32, DType::F16)
        .expect("FP32 accumulation and FP16 convolution")
        .with_staging_capacity(2);
    let mut state = TensorDecodeState::<TestBackend>::new(config, 2, &device()).unwrap();
    state.main_mut().add_recurrent(spec).unwrap();
    state
        .main_mut()
        .recurrent_mut(0)
        .unwrap()
        .set_batch_state(
            0,
            tensor4(vec![1.0, 2.0], [2, 1, 1, 1]),
            tensor3(vec![3.0, 4.0, 5.0, 6.0], [2, 1, 2]).cast(DType::F16),
        )
        .unwrap();
    state.advance(&[1, 1]).unwrap();
    let mark = state.begin_transaction().unwrap();
    state
        .main_mut()
        .recurrent_mut(0)
        .unwrap()
        .stage_batch_snapshots(
            0,
            tensor5(vec![11.0, 12.0], [2, 1, 1, 1, 1]),
            tensor4(vec![13.0, 14.0, 15.0, 16.0], [2, 1, 1, 2]).cast(DType::F16),
        )
        .unwrap();
    state.advance(&[1, 1]).unwrap();
    let rows = if whole_batch { 2 } else { 1 };
    let names = if whole_batch {
        [
            "recurrent matrix snapshot batch",
            "convolution snapshot batch",
        ]
    } else {
        ["recurrent matrix snapshots", "convolution snapshots"]
    };

    // Each invalid call has a nonempty destination proposal. Checking both
    // snapshots and committed tensors catches partial publication, while the
    // next successful append proves that an error did not advance its cursor.
    for invalid in 0..4 {
        let matrix_shape = [rows, 1, if invalid == 0 { 2 } else { 1 }, 1, 1];
        let convolution_shape = [rows, if invalid == 1 { 2 } else { 1 }, 1, 2];
        let matrices = tensor5(vec![90.0; matrix_shape.iter().product()], matrix_shape)
            .cast(if invalid == 2 { DType::F64 } else { DType::F32 });
        let convolutions = tensor4(
            vec![99.0; convolution_shape.iter().product()],
            convolution_shape,
        )
        .cast(if invalid == 3 { DType::F32 } else { DType::F16 });
        let cache = state.main_mut().recurrent_mut(0).unwrap();
        let error = if whole_batch {
            cache.stage_batch_snapshots(0, matrices, convolutions)
        } else {
            cache.stage_sequence_snapshots(0, 1, matrices, convolutions)
        }
        .expect_err("incompatible snapshot must be rejected before publication");
        match (invalid, error) {
            (
                0 | 1,
                TensorCacheError::ShapeMismatch {
                    tensor,
                    expected,
                    actual,
                },
            ) => {
                assert_eq!(tensor, names[invalid]);
                let (expected_shape, actual_shape) = if invalid == 0 {
                    (vec![rows, 1, 1, 1, 1], matrix_shape.to_vec())
                } else {
                    (vec![rows, 1, 1, 2], convolution_shape.to_vec())
                };
                assert_eq!(expected, expected_shape);
                assert_eq!(actual, actual_shape);
            }
            (
                2 | 3,
                TensorCacheError::DTypeMismatch {
                    tensor,
                    expected,
                    actual,
                },
            ) => {
                assert_eq!(tensor, names[invalid - 2]);
                assert_eq!(expected, if invalid == 2 { DType::F32 } else { DType::F16 });
                assert_eq!(actual, if invalid == 2 { DType::F64 } else { DType::F32 });
            }
            (_, error) => panic!("unexpected validation error: {error}"),
        }
        assert_eq!(state.lengths(), &[1, 1]);
        assert_eq!(state.working_lengths(), &[2, 2]);
        let cache = state.main().recurrent(0).unwrap();
        assert_eq!(cache.staged_length(0, 0).unwrap(), 1);
        assert_eq!(cache.staged_length(0, 1).unwrap(), 1);
        let working = cache.read_working_batch(0).unwrap();
        assert_eq!(working.staged_rows(), &[true, true]);
        assert_eq!(working.matrix().dims(), [2, 1, 1, 1]);
        assert_eq!(working.matrix().dtype(), DType::F32);
        assert_eq!(
            working.matrix().to_data().to_vec::<f32>().unwrap(),
            [11.0, 12.0]
        );
        assert_eq!(working.convolution().dims(), [2, 1, 2]);
        assert_eq!(working.convolution().dtype(), DType::F16);
        assert_eq!(
            working
                .convolution()
                .clone()
                .cast(DType::F32)
                .to_data()
                .to_vec::<f32>()
                .unwrap(),
            [13.0, 14.0, 15.0, 16.0]
        );
        assert_eq!(
            cache.matrix(0).unwrap().to_data().to_vec::<f32>().unwrap(),
            [1.0, 2.0]
        );
        assert_eq!(
            cache
                .convolution(0)
                .unwrap()
                .clone()
                .cast(DType::F32)
                .to_data()
                .to_vec::<f32>()
                .unwrap(),
            [3.0, 4.0, 5.0, 6.0]
        );
    }

    let cache = state.main_mut().recurrent_mut(0).unwrap();
    if whole_batch {
        cache
            .stage_batch_snapshots(
                0,
                tensor5(vec![21.0, 22.0], [2, 1, 1, 1, 1]),
                tensor4(vec![23.0, 24.0, 25.0, 26.0], [2, 1, 1, 2]).cast(DType::F16),
            )
            .unwrap();
    } else {
        cache
            .stage_sequence_snapshots(
                0,
                1,
                tensor5(vec![22.0], [1, 1, 1, 1, 1]),
                tensor4(vec![25.0, 26.0], [1, 1, 1, 2]).cast(DType::F16),
            )
            .unwrap();
    }
    state.advance(&[usize::from(whole_batch), 1]).unwrap();
    state
        .commit_prefix(mark, &[usize::from(whole_batch), 2])
        .unwrap();
    assert_eq!(state.lengths(), &[1 + usize::from(whole_batch), 3]);
    let cache = state.main().recurrent(0).unwrap();
    let expected_matrix = [if whole_batch { 11.0 } else { 1.0 }, 22.0];
    let expected_convolution = if whole_batch {
        [13.0, 14.0, 25.0, 26.0]
    } else {
        [3.0, 4.0, 25.0, 26.0]
    };
    assert_eq!(
        cache.matrix(0).unwrap().to_data().to_vec::<f32>().unwrap(),
        expected_matrix
    );
    assert_eq!(
        cache
            .convolution(0)
            .unwrap()
            .clone()
            .cast(DType::F32)
            .to_data()
            .to_vec::<f32>()
            .unwrap(),
        expected_convolution
    );

    // A fresh transaction after the repaired commit has independent storage.
    let mark = state.begin_transaction().unwrap();
    state
        .main_mut()
        .recurrent_mut(0)
        .unwrap()
        .stage_batch_snapshots(
            0,
            tensor5(vec![91.0, 92.0], [2, 1, 1, 1, 1]),
            tensor4(vec![93.0, 94.0, 95.0, 96.0], [2, 1, 1, 2]).cast(DType::F16),
        )
        .unwrap();
    state.advance(&[1, 1]).unwrap();
    state.rollback(mark).unwrap();
    assert_eq!(state.lengths(), &[1 + usize::from(whole_batch), 3]);
    assert_eq!(state.working_lengths(), state.lengths());
    let restored = state
        .main()
        .recurrent(0)
        .unwrap()
        .read_working_batch(0)
        .unwrap();
    assert_eq!(restored.staged_rows(), &[false, false]);
    assert_eq!(
        restored.matrix().to_data().to_vec::<f32>().unwrap(),
        expected_matrix
    );
    assert_eq!(
        restored
            .convolution()
            .clone()
            .cast(DType::F32)
            .to_data()
            .to_vec::<f32>()
            .unwrap(),
        expected_convolution
    );
}

/// Per-row shape/dtype failures preserve every row and permit an exact retry.
#[test]
fn snapshot_contract_sequence_rejection_preserves_pending_and_committed_state() {
    assert_snapshot_contract_rejections_are_atomic(false);
}

/// Whole-batch shape/dtype failures cannot partly publish a matrix or convolution.
#[test]
fn snapshot_contract_batch_rejection_preserves_pending_and_committed_state() {
    assert_snapshot_contract_rejections_are_atomic(true);
}
