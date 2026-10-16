//! Storage-order tests: contiguous Flex projections, unchanged values, and safe fallbacks.
//!
//! Every test compares the crate installer against Burn's own record path, which
//! applies the parameter's mapper to the checkpoint tensor in checkpoint order.
//! Values must agree bit for bit; only the physical layout of the installed Flex
//! tensor may differ, and it is inspected through the Flex primitive's strides.

use burn::{
    backend::{Flex, flex::FlexTensor},
    module::{Module, Param, ParamId},
    nn::{LinearConfig, LinearLayout},
    tensor::{DType, Tensor, TensorData},
};
use proptest::prelude::*;

use super::{prefers_contiguous_matrices, transpose_bytes};
use crate::checkpoint::install_float_parameter;

/// Returns the Flex storage behind a float tensor, for stride inspection.
fn primitive(tensor: Tensor<Flex, 2>) -> FlexTensor {
    tensor.into_primitive().tensor()
}

/// Distinct, exactly representable checkpoint values for an `[rows, columns]` matrix.
fn checkpoint(rows: usize, columns: usize) -> TensorData {
    let values = (0..rows * columns)
        .map(|index| index as f32 - 7.0)
        .collect::<Vec<_>>();
    TensorData::new(values, [rows, columns])
}

/// Loads `data` through Burn's plain record path, the reference for every mapper.
fn burn_reference<const D: usize>(
    parameter: &Param<Tensor<Flex, D>>,
    data: TensorData,
) -> Param<Tensor<Flex, D>> {
    let dtype = data.dtype;
    let tensor = Tensor::<Flex, D>::from_data(data, (&Default::default(), dtype));
    parameter
        .clone()
        .load_record(Param::initialized(ParamId::new(), tensor))
}

/// Builds a lazy column-layout projection with the given widths.
fn column_projection(input: usize, output: usize) -> burn::nn::Linear<Flex> {
    LinearConfig::new(input, output)
        .with_bias(false)
        .with_layout(LinearLayout::Col)
        .init::<Flex>(&Default::default())
}

/// The inference Flex backend prefers contiguous matrices.
#[test]
fn flex_inference_backend_prefers_contiguous_matrices() {
    assert!(prefers_contiguous_matrices::<Flex>(&Default::default()));
}

/// Autodiff keeps checkpoint order so training layouts and costs are unchanged.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_flex_keeps_checkpoint_order() {
    use burn::backend::Autodiff;

    assert!(!prefers_contiguous_matrices::<Autodiff<Flex>>(
        &Default::default()
    ));
    let device = Default::default();
    let mut projection = crate::operations::init_linear::<Autodiff<Flex>>(
        &LinearConfig::new(4, 3)
            .with_bias(false)
            .with_layout(LinearLayout::Col),
        &device,
    );
    install_float_parameter(&mut projection.weight, checkpoint(3, 4)).unwrap();
    let reference = checkpoint(3, 4)
        .to_vec::<f32>()
        .unwrap()
        .chunks(4)
        .map(<[f32]>::to_vec)
        .collect::<Vec<_>>();
    let installed = projection.weight.val().into_data().to_vec::<f32>().unwrap();
    for input in 0..4 {
        for output in 0..3 {
            assert_eq!(installed[input * 3 + output], reference[output][input]);
        }
    }
}

/// A column-layout projection becomes contiguous `[in, out]` with unchanged values.
#[test]
fn column_layout_projection_is_installed_contiguously() {
    for (input, output) in [(5, 3), (3, 5), (4, 4), (33, 70)] {
        let mut projection = column_projection(input, output);
        let reference = burn_reference(&projection.weight, checkpoint(output, input));
        assert_eq!(
            primitive(reference.val()).layout().strides(),
            [1, input as isize],
            "Burn's record path yields a strided view"
        );

        install_float_parameter(&mut projection.weight, checkpoint(output, input)).unwrap();

        let installed = primitive(projection.weight.val());
        assert!(installed.is_contiguous());
        assert_eq!(installed.layout().strides(), [output as isize, 1]);
        assert_eq!(projection.weight.val().dims(), [input, output]);
        assert_eq!(
            projection.weight.val().into_data(),
            reference.val().into_data()
        );
        assert_eq!(
            projection.weight.clone().into_record().to_data(),
            checkpoint(output, input),
            "the save mapper still emits the checkpoint matrix"
        );
    }
}

/// Projections through contiguous and strided storage agree exactly on exact inputs.
#[test]
fn contiguous_and_strided_projections_agree() {
    let (input, output) = (6, 9);
    let mut projection = column_projection(input, output);
    let mut strided = projection.clone();
    strided.weight = burn_reference(&strided.weight, checkpoint(output, input));
    install_float_parameter(&mut projection.weight, checkpoint(output, input)).unwrap();

    let activations = Tensor::<Flex, 3>::from_data(
        TensorData::new(
            (0..2 * 3 * input)
                .map(|index| (index % 5) as f32 - 2.0)
                .collect::<Vec<_>>(),
            [2, 3, input],
        ),
        &Default::default(),
    );
    let contiguous = crate::operations::linear(activations.clone(), &projection).unwrap();
    let reference = crate::operations::linear(activations, &strided).unwrap();
    assert_eq!(contiguous.into_data(), reference.into_data());
}

/// Every float storage dtype keeps its exact bits and gains the contiguous layout.
#[test]
fn every_float_dtype_is_reordered_bit_exactly() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut projection = column_projection(7, 3);
        let data = checkpoint(3, 7).convert_dtype(dtype);
        let reference = burn_reference(&projection.weight, data.clone());

        install_float_parameter(&mut projection.weight, data).unwrap();

        let installed = projection.weight.val();
        assert_eq!(installed.dtype(), dtype);
        assert!(primitive(installed.clone()).is_contiguous(), "{dtype:?}");
        assert_eq!(installed.into_data(), reference.val().into_data());
    }
}

/// Row-layout (identity-mapper) matrices keep checkpoint order and contiguity.
#[test]
fn identity_mapper_matrix_keeps_checkpoint_order() {
    let mut projection = LinearConfig::new(4, 6)
        .with_bias(false)
        .init::<Flex>(&Default::default());
    install_float_parameter(&mut projection.weight, checkpoint(4, 6)).unwrap();
    let installed = primitive(projection.weight.val());
    assert_eq!(installed.layout().strides(), [6, 1]);
    assert_eq!(projection.weight.val().into_data(), checkpoint(4, 6));
}

/// Single-row or single-column matrices need no reordering and keep exact values.
#[test]
fn degenerate_matrices_keep_checkpoint_order() {
    for (input, output) in [(1, 5), (5, 1), (1, 1)] {
        let mut projection = column_projection(input, output);
        let reference = burn_reference(&projection.weight, checkpoint(output, input));
        install_float_parameter(&mut projection.weight, checkpoint(output, input)).unwrap();
        assert_eq!(
            projection.weight.val().into_data(),
            reference.val().into_data()
        );
    }
}

/// A mapper that transposes and scales still receives the checkpoint's exact values.
#[test]
fn arithmetic_mapper_sees_unchanged_logical_values() {
    let device = Default::default();
    let parameter = Param::uninitialized(
        ParamId::new(),
        |device, _| Tensor::<Flex, 2>::zeros([3, 5], device),
        device,
        false,
        [3, 5].into(),
    )
    .load_mapper(|tensor| tensor.transpose() * 2.0);
    let reference = burn_reference(&parameter, checkpoint(3, 5));
    let mut installed = parameter;

    install_float_parameter(&mut installed, checkpoint(3, 5)).unwrap();

    assert_eq!(installed.val().dims(), [5, 3]);
    assert_eq!(installed.val().into_data(), reference.val().into_data());
}

/// A shape-specific mapper panics on the probe, yet the install still succeeds exactly.
#[test]
fn shape_specific_mapper_falls_back_to_checkpoint_order() {
    let device = Default::default();
    let parameter = Param::uninitialized(
        ParamId::new(),
        |device, _| Tensor::<Flex, 2>::zeros([4, 3], device),
        device,
        false,
        [4, 3].into(),
    )
    .load_mapper(|tensor| tensor.reshape([3, 4]));
    let reference = burn_reference(&parameter, checkpoint(4, 3));
    let mut installed = parameter;

    install_float_parameter(&mut installed, checkpoint(4, 3)).unwrap();

    assert_eq!(installed.val().dims(), [3, 4]);
    assert_eq!(installed.val().into_data(), reference.val().into_data());
}

/// The probe leaves a lazy parameter uninitialized when installation fails later.
#[test]
fn rejected_install_never_initializes_or_probes_into_the_parameter() {
    let mut projection = column_projection(4, 3);
    let error = install_float_parameter(&mut projection.weight, checkpoint(4, 3)).unwrap_err();
    assert!(error.to_string().contains("shape"));
    assert!(!projection.weight.is_initialized());
}

/// Reinstalling into an initialized column-layout weight replaces it contiguously.
#[test]
fn reinstall_into_initialized_projection_stays_contiguous() {
    let mut projection = column_projection(5, 2);
    install_float_parameter(&mut projection.weight, checkpoint(2, 5)).unwrap();
    let replacement = TensorData::new(
        (0..10).map(|value| value as f32 * 0.5).collect::<Vec<_>>(),
        [2, 5],
    );
    let reference = burn_reference(&projection.weight, replacement.clone());

    install_float_parameter(&mut projection.weight, replacement).unwrap();

    assert!(primitive(projection.weight.val()).is_contiguous());
    assert_eq!(
        projection.weight.val().into_data(),
        reference.val().into_data()
    );
}

/// Rank-3 parameters are outside the matrix policy and keep checkpoint order.
#[test]
fn higher_rank_parameters_keep_checkpoint_order() {
    let device = Default::default();
    let mut parameter = Param::uninitialized(
        ParamId::new(),
        |device, _| Tensor::<Flex, 3>::zeros([2, 3, 4], device),
        device,
        false,
        [2, 3, 4].into(),
    )
    .load_mapper(|tensor| tensor.swap_dims(1, 2));
    let data = TensorData::new(
        (0..24).map(|value| value as f32).collect::<Vec<_>>(),
        [2, 3, 4],
    );
    let reference = burn_reference(&parameter, data.clone());

    install_float_parameter(&mut parameter, data).unwrap();

    assert_eq!(parameter.val().into_data(), reference.val().into_data());
}

/// Byte transposition rejects unsupported widths and mismatched buffer lengths.
#[test]
fn byte_transpose_rejects_unsupported_widths_and_lengths() {
    assert_eq!(transpose_bytes(&[0; 6], 2, 3, 1), None);
    assert_eq!(transpose_bytes(&[0; 7], 2, 3, 1), None);
    assert_eq!(transpose_bytes(&[0; 11], 2, 3, 2), None);
    assert_eq!(transpose_bytes(&[0; 16], 2, 3, 2), None);
    assert_eq!(transpose_bytes(&[0; 24], usize::MAX, 2, 4), None);
    assert_eq!(transpose_bytes(&[], 0, 3, 4), Some(Vec::new()));
}

/// A transpose above the parallel threshold (ragged final band) places every element.
///
/// This runs with the host's own parallelism; the multi-band path is checked
/// for explicit thread budgets, independent of the host, in the conversion
/// kernel's tests (`threaded_index_transpose_places_every_element_for_every_budget`).
#[test]
fn parallel_byte_transpose_matches_naive_reference() {
    let (rows, columns) = (1031, 1037);
    let source = (0..rows * columns)
        .flat_map(|index| u32::try_from(index).unwrap().to_ne_bytes())
        .collect::<Vec<_>>();
    let transposed = transpose_bytes(&source, rows, columns, 4).unwrap();
    assert_eq!(transposed.len(), source.len());
    for (target, scalar) in transposed.as_chunks::<4>().0.iter().enumerate() {
        let (column, row) = (target / rows, target % rows);
        let value = u32::from_ne_bytes(*scalar);
        assert_eq!(value as usize, row * columns + column, "target {target}");
    }
}

proptest! {
    /// Transposing any matrix of 2-, 4- or 8-byte scalars moves every scalar exactly once.
    #[test]
    fn byte_transpose_is_an_exact_permutation(
        rows in 0_usize..70,
        columns in 0_usize..70,
        width in prop::sample::select(vec![2_usize, 4, 8]),
        seed in any::<u8>(),
    ) {
        let source = (0..rows * columns * width)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed))
            .collect::<Vec<_>>();
        let transposed = transpose_bytes(&source, rows, columns, width).unwrap();
        prop_assert_eq!(transposed.len(), source.len());
        for row in 0..rows {
            for column in 0..columns {
                let from = (row * columns + column) * width;
                let to = (column * rows + row) * width;
                prop_assert_eq!(&transposed[to..to + width], &source[from..from + width]);
            }
        }
        let restored = transpose_bytes(&transposed, columns, rows, width).unwrap();
        prop_assert_eq!(restored, source);
    }
}
