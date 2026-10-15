//! Device realization of compact-batch maps against host expectations.

use burn::{backend::Flex, tensor::TensorData};

use super::*;

/// CPU Flex backend with F32 values and I32 indices.
type TestBackend = Flex<f32, i32>;

/// Ragged fixture: left padding, right padding, an interior hole, and an inactive row.
fn ragged() -> (Vec<Vec<usize>>, CompactBatch<TestBackend>) {
    let columns = vec![vec![2, 3], vec![0, 1], vec![0, 2], vec![]];
    let batch = CompactBatch::new(&columns, 4, &Default::default()).unwrap();
    (columns, batch)
}

/// Rectangular `[4, 4, 2]` values whose entry `(b, s, d)` encodes its coordinates.
fn coordinates() -> Tensor<TestBackend, 3> {
    let values = (0..32).map(|index| index as f32 + 1.0).collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [4, 4, 2]), &Default::default())
}

/// Token compaction gathers valid columns and replaces padding with token zero.
#[test]
fn compact_tokens_gathers_valid_columns() {
    let (_, batch) = ragged();
    let tokens = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(
            vec![-1, -1, 7, 8, 9, 10, 99, 99, 11, -5, 12, -5, 99, 99, 99, 99],
            [4, 4],
        ),
        &Default::default(),
    );
    let compact = batch.compact_tokens(tokens).unwrap();
    assert_eq!(
        compact
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .unwrap(),
        [7, 8, 9, 10, 11, 12, 0, 0]
    );
}

/// Compaction followed by scatter restores valid columns and zeros padding.
#[test]
fn compact_then_rectangular_round_trips_valid_columns() {
    let (columns, batch) = ragged();
    let rectangular = coordinates();
    let flat = rectangular.clone().reshape([16, 2]);
    let gather =
        index_tensor::<TestBackend>(&batch.plan().gather, &Default::default(), "test").unwrap();
    let compact = flat.select(0, gather).reshape([4, 2, 2]);
    let restored = batch.to_rectangular(compact.clone()).unwrap();
    let original = rectangular.into_data().to_vec::<f32>().unwrap();
    let restored = restored.into_data().to_vec::<f32>().unwrap();
    for (row, valid) in columns.iter().enumerate() {
        for column in 0..4 {
            for channel in 0..2 {
                let index = (row * 4 + column) * 2 + channel;
                let expected = if valid.contains(&column) {
                    original[index]
                } else {
                    0.0
                };
                assert_eq!(restored[index], expected);
            }
        }
    }
    // Last tokens: each row's final valid column, zero for the inactive row.
    let last = batch.last_tokens(compact).unwrap();
    assert_eq!(last.dims(), [4, 1, 2]);
    assert_eq!(
        last.into_data().to_vec::<f32>().unwrap(),
        [7.0, 8.0, 11.0, 12.0, 21.0, 22.0, 0.0, 0.0]
    );
}

/// Packing keeps exactly the valid tokens and unpacking zeros the padding.
#[test]
fn pack_and_unpack_are_inverse_on_valid_slots() {
    let (_, batch) = ragged();
    let compact = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(
            (1..=8).map(|value| value as f32).collect::<Vec<_>>(),
            [4, 2, 1],
        ),
        &Default::default(),
    );
    let packed = batch.pack(compact).unwrap();
    assert_eq!(
        packed.clone().into_data().to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let unpacked = batch.unpack(packed).unwrap();
    assert_eq!(
        unpacked.into_data().to_vec::<f32>().unwrap(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]
    );
    assert!(
        batch
            .unpack(Tensor::<TestBackend, 2>::zeros([5, 1], &Default::default()))
            .is_err()
    );
}

/// Integer router selections scatter like floating values.
#[test]
fn integer_values_scatter_with_zero_padding() {
    let (_, batch) = ragged();
    let packed = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1, 2, 3, 4, 5, 6], [6, 1]),
        &Default::default(),
    );
    let rectangular = batch.to_rectangular(batch.unpack(packed).unwrap()).unwrap();
    assert_eq!(
        rectangular
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .unwrap(),
        [0, 0, 1, 2, 3, 4, 0, 0, 5, 0, 6, 0, 0, 0, 0, 0]
    );
}

/// Identity batches pass tensors through untouched.
#[test]
fn identity_batch_skips_every_map() {
    let batch = CompactBatch::<TestBackend>::new(&[vec![0, 1], vec![0, 1]], 2, &Default::default())
        .unwrap();
    assert!(batch.plan().is_identity());
    let values = Tensor::<TestBackend, 3>::ones([2, 2, 3], &Default::default());
    assert_eq!(
        batch.to_rectangular(values.clone()).unwrap().into_data(),
        values.clone().into_data()
    );
    assert_eq!(batch.pack(values.clone()).unwrap().dims(), [4, 3]);
    assert_eq!(batch.last_tokens(values).unwrap().dims(), [2, 1, 3]);
    assert_eq!(
        batch
            .query_positions(&[5, 0])
            .unwrap()
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .unwrap(),
        [5, 6, 0, 1]
    );
}

/// Mis-sized inputs and invalid plans are contextual errors.
#[test]
fn invalid_inputs_are_rejected() {
    let device = Default::default();
    assert!(CompactBatch::<TestBackend>::new(&[vec![1, 0]], 2, &device).is_err());
    let (_, batch) = ragged();
    assert!(
        batch
            .compact_tokens(Tensor::<TestBackend, 2, Int>::zeros([4, 3], &device))
            .is_err()
    );
    assert!(
        batch
            .to_rectangular(Tensor::<TestBackend, 3>::zeros([4, 3, 1], &device))
            .is_err()
    );
    let empty = CompactBatch::<TestBackend>::new(&[vec![]], 2, &device).unwrap();
    assert!(
        empty
            .to_rectangular(Tensor::<TestBackend, 3>::zeros([1, 0, 1], &device))
            .is_err()
    );
}
