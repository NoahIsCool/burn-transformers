//! Training boundaries for packed checkpoint weights on Burn 0.21.
//!
//! These tests intentionally distinguish packed inference from optimization
//! through an explicitly dequantized, frozen downstream operation.

use burn::{
    backend::{Autodiff, Flex},
    module::{Module, Param},
    nn::Linear,
    optim::{GradientsParams, Optimizer, SgdConfig},
    tensor::{DType, Tensor, TensorData, backend::Backend},
};

use crate::{
    attention::{AttentionPrecisionMode, GroupedQueryAttentionConfig},
    checkpoint::{
        ParameterInstallError, install_float_parameter,
        quantization::{
            QuantizationGroupSize, SymmetricQuantization, WeightQuantizationConfig,
            quantize_tensor_data,
        },
    },
    intervention::{Hooks, Layer},
    operations::linear,
};

/// Differentiable CPU backend used to exercise packed-weight training boundaries.
type Ad = Autodiff<Flex>;

/// Quantize a 32-by-32 identity matrix into the requested symmetric packed format.
fn packed_identity(scheme: SymmetricQuantization) -> TensorData {
    let values = (0..32 * 32)
        .map(|index| if index / 32 == index % 32 { 1.0 } else { 0.0 })
        .collect::<Vec<f32>>();
    quantize_tensor_data(
        &TensorData::new(values, [32, 32]),
        WeightQuantizationConfig {
            scheme,
            group_size: QuantizationGroupSize::new(32).unwrap(),
        },
    )
    .unwrap()
    .into_burn_qfloat_data()
    .unwrap()
}

/// Trainable inserted projection followed by an independently frozen linear suffix.
#[derive(Module, Debug)]
struct AdapterExperiment<B: Backend> {
    /// Trainable adapter whose weights should receive optimization updates.
    adapter: Linear<B>,
    /// Frozen suffix, explicitly expanded from packed checkpoint data.
    downstream: Linear<B>,
}

/// A trainable module inserted by a hook learns through a frozen suffix. The
/// suffix came from packed Q4/Q8 data but uses explicitly expanded F32 storage.
#[test]
fn optimizer_learns_through_explicitly_dequantized_frozen_q4_and_q8_suffix() {
    let device = Default::default();
    for scheme in [SymmetricQuantization::Q4, SymmetricQuantization::Q8] {
        let packed = Tensor::<Flex, 2>::from_data(packed_identity(scheme), &device);
        assert!(matches!(packed.dtype(), DType::QFloat(_)));
        // Dequantize before lifting into autodiff. Burn's autodiff dequantize
        // operator is not implemented, even for a frozen tensor.
        let floating = Tensor::<Ad, 2>::from_inner(packed.dequantize());
        assert_eq!(floating.dtype(), DType::F32);
        let mut model = AdapterExperiment {
            adapter: Linear {
                weight: Param::from_tensor(Tensor::<Ad, 2>::zeros([32, 32], &device)),
                bias: None,
            },
            downstream: Linear {
                weight: Param::from_tensor(floating),
                bias: None,
            }
            .no_grad(),
        };
        let original_suffix = model.downstream.weight.val().to_data();
        let original_adapter = model.adapter.weight.val().to_data();
        let mut source = vec![0.0_f32; 4 * 32];
        source[0] = 1.0;
        source[32] = -1.0;
        source[65] = 1.0;
        source[97] = -1.0;
        let source = Tensor::<Ad, 3>::from_data(TensorData::new(source, [4, 1, 32]), &device);
        let target = source.clone() * 2.0;
        let mut optimizer = SgdConfig::new().init();
        let point = Layer::decoder(0).residual_post();
        let mut initial_loss = 0.0_f32;
        let mut final_loss = 0.0_f32;
        for step in 0..80 {
            let mut hooks = Hooks::new();
            hooks.try_edit(&point, |hidden| linear(hidden, &model.adapter));
            let adapted = hooks
                .run(vec![point.info()], |hooks| {
                    hooks.apply(point.clone(), source.clone())
                })
                .unwrap();
            let prediction = linear(adapted, &model.downstream).unwrap();
            let loss = (prediction - target.clone()).square().sum() / 4.0;
            let value = loss.clone().into_scalar();
            if step == 0 {
                initial_loss = value;
            }
            final_loss = value;
            let gradients = loss.backward();
            if step == 0 {
                assert!(model.downstream.weight.val().grad(&gradients).is_none());
                let adapter_gradient = model.adapter.weight.val().grad(&gradients).unwrap();
                assert!(adapter_gradient.abs().sum().into_scalar() > 0.0);
            }
            let gradients = GradientsParams::from_grads(gradients, &model);
            drop(hooks);
            model = optimizer.step(0.1, model, gradients);
        }
        assert!(
            initial_loss > 1.0,
            "nontrivial starting loss: {initial_loss}"
        );
        assert!(
            final_loss < 1e-6,
            "{scheme:?}: {initial_loss} -> {final_loss}"
        );
        assert_eq!(model.downstream.weight.val().to_data(), original_suffix);
        assert_ne!(model.adapter.weight.val().to_data(), original_adapter);
        assert!(!model.downstream.weight.val().is_require_grad());
    }
}

/// Freezing cannot make Burn's unimplemented packed autodiff matmul usable,
/// whether or not the activation itself currently requests a gradient.
#[test]
fn packed_autodiff_projection_is_rejected_before_dispatch() {
    let device = Default::default();
    for scheme in [SymmetricQuantization::Q4, SymmetricQuantization::Q8] {
        let packed = Tensor::<Flex, 2>::from_data(packed_identity(scheme), &device);
        // Lifting an existing QFloat primitive itself is supported; executing
        // Autodiff's QFloat matmul would invoke its unimplemented dequantize.
        let projection = Linear {
            weight: Param::from_tensor(Tensor::<Ad, 2>::from_inner(packed)),
            bias: None,
        }
        .no_grad();
        for require_grad in [false, true] {
            let input = Tensor::<Ad, 2>::ones([2, 32], &device).set_require_grad(require_grad);
            let result = linear(input, &projection);
            assert!(
                matches!(result, Err(crate::Error::BackendCapability(reason))
                if reason.contains("input gradients through frozen weights"))
            );
            assert!(matches!(projection.weight.val().dtype(), DType::QFloat(_)));
        }
    }
}

/// Packed Q/K/V and output weights share the projection capability boundary,
/// independently of which dense attention graph the experiment selects.
#[test]
fn packed_autodiff_attention_projections_return_capability_errors() {
    let device = Default::default();
    for scheme in [SymmetricQuantization::Q4, SymmetricQuantization::Q8] {
        for precision in [
            AttentionPrecisionMode::BackendDefault,
            AttentionPrecisionMode::EagerFp32Softmax,
        ] {
            for projection_index in 0..4 {
                let packed = Tensor::<Flex, 2>::from_data(packed_identity(scheme), &device);
                let packed = Linear {
                    weight: Param::from_tensor(Tensor::<Ad, 2>::from_inner(packed)),
                    bias: None,
                }
                .no_grad();
                let mut attention = GroupedQueryAttentionConfig::new(32, 1, 1, 32)
                    .with_precision_mode(precision)
                    .init::<Ad>(&device)
                    .unwrap()
                    .no_grad();
                match projection_index {
                    0 => attention.q_proj = packed,
                    1 => attention.k_proj = packed,
                    2 => attention.v_proj = packed,
                    _ => attention.o_proj = packed,
                }
                let hidden = Tensor::<Ad, 3>::ones([1, 2, 32], &device).require_grad();
                let result = attention
                    .project_self(hidden)
                    .and_then(|projection| attention.attend(projection, None, None, false));
                assert!(
                    matches!(result, Err(crate::Error::BackendCapability(reason))
                    if reason.contains("input gradients through frozen weights"))
                );
            }
        }
    }
}

/// Parameter installation rejects packed data without initializing or replacing
/// the destination; otherwise Burn reaches `Autodiff::q_from_data`'s todo!.
#[test]
fn packed_autodiff_checkpoint_install_is_rejected_before_initialization() {
    let device = burn::backend::flex::FlexDevice;
    for scheme in [SymmetricQuantization::Q4, SymmetricQuantization::Q8] {
        for frozen in [false, true] {
            let mut parameter = Param::<Tensor<Ad, 2>>::uninitialized(
                burn::module::ParamId::new(),
                |_, _| panic!("rejected packed load must not initialize the target"),
                device,
                !frozen,
                [32, 32].into(),
            );
            let id = parameter.id;
            let result = install_float_parameter(&mut parameter, packed_identity(scheme));
            assert_eq!(result, Err(ParameterInstallError::QuantizedAutodiff));
            assert!(!parameter.is_initialized());
            assert_eq!(parameter.id, id);
        }
    }
}
