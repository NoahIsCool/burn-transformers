use super::*;
use burn::{backend::Flex, tensor::TensorData};

type Cpu = Flex<f32, i32>;

#[test]
fn large_vocabulary_loss_matches_f64_despite_sequential_f32_tail_loss() {
    let device = Default::default();
    let classes = 50_303;
    let mut values = vec![-16.0_f32; classes];
    values[0] = 0.0;
    let logits = Tensor::<Cpu, 2>::from_data(TensorData::new(values, [1, classes]), &device);
    let targets = Tensor::from_ints([0], &device);
    let expected = (1.0 + (classes - 1) as f64 * (-16.0_f64).exp()).ln();
    let stable = cross_entropy(logits, targets).unwrap().into_scalar();
    // A deterministic negative control for scalar F32 accumulation. Native
    // Burn may instead use SIMD reductions (the `train` feature enables them),
    // so this test must not require native loss to remain inaccurate.
    let tail = (-16.0_f32).exp();
    let sequential_normalizer = (1..classes).fold(1.0_f32, |sum, _| sum + tail);
    let sequential = sequential_normalizer.ln();
    assert!((f64::from(stable) - expected).abs() < 2e-7);
    assert!((f64::from(sequential) - expected).abs() > 1e-4);
}

#[test]
fn arbitrary_axis_odd_width_and_f64_precision_are_preserved() {
    let device = Default::default();
    let input = Tensor::<Cpu, 3>::from_data([[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]], &device)
        .cast(burn::tensor::FloatDType::F64);
    let output = log_softmax(input, 1).unwrap();
    assert_eq!(output.dtype(), DType::F64);
    assert_eq!(output.dims(), [1, 3, 2]);
    let values = output.into_data().to_vec::<f64>().unwrap();
    let norm = (1.0_f64 + 1.0_f64.exp() + 2.0_f64.exp()).ln();
    for (actual, expected) in
        values
            .into_iter()
            .zip([-norm, -norm, 1.0 - norm, 1.0 - norm, 2.0 - norm, 2.0 - norm])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn loss_contract_rejects_invalid_axes_labels_shapes_and_dtypes() {
    let device = Default::default();
    let input = Tensor::<Cpu, 2>::zeros([2, 3], &device);
    assert!(log_softmax(input.clone(), 2).is_err());
    assert!(cross_entropy(input.clone(), Tensor::from_ints([0], &device)).is_err());
    assert!(cross_entropy(input.clone(), Tensor::from_ints([-1, 0], &device)).is_err());
    assert!(cross_entropy(input.clone(), Tensor::from_ints([0, 3], &device)).is_err());
    assert!(log_softmax(input.clone().cast(burn::tensor::FloatDType::F16), 1).is_err());
    assert!(kl_divergence(input.clone(), Tensor::zeros([2, 4], &device)).is_err());
    assert!(kl_divergence(input.clone(), input.cast(burn::tensor::FloatDType::F64)).is_err());
}

#[test]
fn kl_uses_teacher_to_student_direction_and_example_mean() {
    let device = Default::default();
    let student = Tensor::<Cpu, 2>::from_data([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device);
    let teacher = Tensor::from_data([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]], &device);
    let norm = 1.0_f64.exp() + 1.0 + (-1.0_f64).exp();
    let expected = [1.0_f64, 0.0, -1.0]
        .into_iter()
        .map(|x| {
            let p = x.exp() / norm;
            p * (p.ln() + 3.0_f64.ln())
        })
        .sum::<f64>();
    let loss = kl_divergence(student, teacher.clone())
        .unwrap()
        .into_scalar();
    assert!((f64::from(loss) - expected).abs() < 2e-7);
    assert_eq!(
        kl_divergence(teacher.clone(), teacher)
            .unwrap()
            .into_scalar(),
        0.0
    );
}

#[cfg(feature = "autodiff")]
mod autodiff {
    use super::*;
    use crate::{
        intervention::{Hooks, Layer},
        operations,
    };
    use burn::{
        backend::Autodiff,
        module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId},
        nn::{Linear, LinearConfig, LinearLayout},
        optim::{AdamConfig, GradientsParams, Optimizer, SgdConfig},
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
        tensor::backend::Backend,
    };
    type Ad = Autodiff<Cpu>;

    #[test]
    fn stable_cross_entropy_gradient_matches_probability_minus_label() {
        let device = Default::default();
        let logits =
            Tensor::<Ad, 2>::from_data([[1.0, 2.0, 3.0], [2.0, -1.0, 0.0]], &device).require_grad();
        let loss = cross_entropy(logits.clone(), Tensor::from_ints([2, 0], &device)).unwrap();
        let gradient = logits
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (index, (row, target)) in [[1.0_f64, 2.0, 3.0], [2.0, -1.0, 0.0]]
            .into_iter()
            .zip([2, 0])
            .enumerate()
        {
            let normalizer: f64 = row.into_iter().map(f64::exp).sum();
            for (column, value) in row.into_iter().enumerate() {
                let expected = (value.exp() / normalizer - f64::from(column == target)) / 2.0;
                assert!((f64::from(gradient[index * 3 + column]) - expected).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn kl_teacher_connectivity_and_explicit_loss_cast_remain_differentiable() {
        let device = Default::default();
        let student = Tensor::<Ad, 2>::from_data([[0.5, -0.5, 1.0]], &device).require_grad();
        let teacher = Tensor::<Ad, 2>::from_data([[1.0, 0.0, -1.0]], &device).require_grad();
        let grads = kl_divergence(student.clone(), teacher.clone())
            .unwrap()
            .backward();
        let student_gradient = student
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let teacher_gradient = teacher
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let probabilities = |values: [f64; 3]| {
            let normalizer: f64 = values.into_iter().map(f64::exp).sum();
            values.map(|value| value.exp() / normalizer)
        };
        let p = probabilities([0.5, -0.5, 1.0]);
        let q = probabilities([1.0, 0.0, -1.0]);
        let divergence: f64 = (0..3).map(|i| q[i] * (q[i].ln() - p[i].ln())).sum();
        for i in 0..3 {
            assert!((f64::from(student_gradient[i]) - (p[i] - q[i])).abs() < 1e-7);
            let expected = q[i] * (q[i].ln() - p[i].ln() - divergence);
            assert!((f64::from(teacher_gradient[i]) - expected).abs() < 1e-7);
        }
        // Each backward consumes its graph. Build independent leaves for the
        // separate fixed-teacher experiment, just as a new forward step does.
        let student = student.detach().require_grad();
        let teacher = teacher.detach().require_grad();
        let grads = kl_divergence(student.clone(), teacher.clone().detach())
            .unwrap()
            .backward();
        assert!(student.grad(&grads).is_some());
        assert!(teacher.grad(&grads).is_none());
        let half = Tensor::<Ad, 2>::from_data([[1.0, 0.0]], &device)
            .cast(burn::tensor::FloatDType::F16)
            .detach()
            .require_grad();
        let loss = cross_entropy(
            half.clone().cast(burn::tensor::FloatDType::F32),
            Tensor::from_ints([0], &device),
        )
        .unwrap();
        assert!(half.grad(&loss.backward()).is_some());
    }

    #[derive(Module, Debug)]
    struct EditedModel<B: Backend> {
        adapter: Linear<B>,
        suffix: Linear<B>,
    }

    impl<B: Backend> EditedModel<B> {
        fn init(device: &B::Device) -> Self {
            let mut adapter = LinearConfig::new(2, 2).with_bias(false).init(device);
            adapter.weight = Param::from_tensor(Tensor::zeros([2, 2], device));
            let mut suffix = LinearConfig::new(2, 2).with_bias(false).init(device);
            suffix.weight =
                Param::from_tensor(Tensor::from_data([[1.0, 0.25], [-0.25, 1.0]], device));
            Self {
                adapter,
                suffix: suffix.no_grad(),
            }
        }

        fn forward(&self, input: Tensor<B, 3>, auxiliary: bool) -> Tensor<B, 1> {
            let point = Layer::decoder(0).residual_post();
            let mut hooks = Hooks::new();
            hooks.edit(point.clone().token(0), |hidden| {
                self.adapter.forward(hidden)
            });
            let captured = hooks.capture_with_grad(point.clone().token(0));
            let detached = hooks.capture(point.clone().token(0));
            let output = hooks
                .run(vec![point.info()], |hooks| {
                    let hidden = hooks.apply(point, input)?;
                    operations::linear(hidden, &self.suffix)
                })
                .unwrap();
            assert!(!hooks.get(&detached).unwrap().is_require_grad());
            let labels = Tensor::from_ints([0, 1, 0, 1], &output.device());
            let loss = cross_entropy(output.reshape([4, 2]), labels).unwrap();
            if auxiliary {
                loss + hooks.get(&captured).unwrap().square().mean() * 0.001
            } else {
                loss
            }
        }
    }

    fn input<B: Backend>(device: &B::Device) -> Tensor<B, 3> {
        Tensor::from_data(
            [[[1.0, 0.0]], [[0.0, 1.0]], [[2.0, 0.0]], [[0.0, 2.0]]],
            device,
        )
    }

    #[test]
    fn optimizer_learns_through_selected_hook_and_frozen_downstream_module() {
        let device = Default::default();
        let mut model = EditedModel::<Ad>::init(&device);
        let frozen = model.suffix.weight.val().into_data();
        let initial = model.valid().forward(input(&device), false).into_scalar();
        let mut optimizer = AdamConfig::new().init();
        for _ in 0..30 {
            let loss = model.forward(input(&device), true);
            let grads = loss.backward();
            assert!(model.adapter.weight.val().grad(&grads).is_some());
            assert!(model.suffix.weight.val().grad(&grads).is_none());
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(0.05, model, grads);
            assert_eq!(model.suffix.weight.val().into_data(), frozen);
        }
        let final_loss = model.valid().forward(input(&device), false).into_scalar();
        assert!(final_loss < initial * 0.2, "{initial} -> {final_loss}");
    }

    #[test]
    fn records_restore_parameter_ids_optimizer_moments_and_next_update_exactly() {
        let device = Default::default();
        let mut model = EditedModel::<Ad>::init(&device);
        let round_trip = model.valid().train::<Ad>();
        assert!(round_trip.adapter.weight.val().is_require_grad());
        assert!(!round_trip.suffix.weight.val().is_require_grad());
        assert_eq!(round_trip.adapter.weight.id, model.adapter.weight.id);
        let mut optimizer = AdamConfig::new().init();
        for _ in 0..4 {
            let grads =
                GradientsParams::from_grads(model.forward(input(&device), true).backward(), &model);
            model = optimizer.step(0.03, model, grads);
        }
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let model_bytes = recorder.record(model.clone().into_record(), ()).unwrap();
        let optimizer_bytes = recorder.record(optimizer.to_record(), ()).unwrap();
        let mut restored = EditedModel::<Ad>::init(&device)
            .load_record(recorder.load(model_bytes.clone(), &device).unwrap());
        let mut restored_optimizer = AdamConfig::new()
            .init()
            .load_record(recorder.load(optimizer_bytes, &device).unwrap());
        assert_eq!(restored.adapter.weight.id, model.adapter.weight.id);
        assert!(!restored.suffix.weight.val().is_require_grad());
        let current_grads =
            GradientsParams::from_grads(model.forward(input(&device), true).backward(), &model);
        let restored_grads = GradientsParams::from_grads(
            restored.forward(input(&device), true).backward(),
            &restored,
        );
        model = optimizer.step(0.03, model, current_grads);
        restored = restored_optimizer.step(0.03, restored, restored_grads);
        assert_eq!(
            model.adapter.weight.val().into_data(),
            restored.adapter.weight.val().into_data()
        );
        assert_eq!(
            model.suffix.weight.val().into_data(),
            restored.suffix.weight.val().into_data()
        );

        // Native records intentionally follow the destination's trainability.
        struct Enable;
        impl ModuleMapper<Ad> for Enable {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<Ad, D>>,
            ) -> Param<Tensor<Ad, D>> {
                param.map(|value| value.set_require_grad(true))
            }
        }
        let trainable = EditedModel::<Ad>::init(&device)
            .map(&mut Enable)
            .load_record(recorder.load(model_bytes, &device).unwrap());
        assert!(trainable.suffix.weight.val().is_require_grad());
    }

    #[test]
    fn capture_only_intermediate_loss_trains_the_owned_inserted_module() {
        let device = Default::default();
        let mut model = EditedModel::<Ad>::init(&device);
        let mut optimizer = AdamConfig::new().init();
        let mut first_loss = 0.0;
        let mut final_loss = 0.0;
        for step in 0..30 {
            let point = Layer::decoder(0).residual_post();
            let mut hooks = Hooks::new();
            hooks.edit(point.clone().token(0), |hidden| {
                model.adapter.forward(hidden)
            });
            let features = hooks.capture_with_grad(point.clone());
            let _unused_output = hooks
                .run(vec![point.info()], |hooks| {
                    hooks.apply(point, input(&device))
                })
                .unwrap();
            let loss = (hooks.get(&features).unwrap() - 1.0).square().mean();
            let value = loss.clone().into_scalar();
            if step == 0 {
                first_loss = value;
            }
            final_loss = value;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            drop(hooks);
            model = optimizer.step(0.05, model, grads);
        }
        assert!(final_loss < first_loss * 0.2);
    }

    #[test]
    fn staged_t5_execution_preserves_every_parameter_gradient() {
        use crate::{
            CausalLmInput, ForwardOptions,
            models::t5::{T5Config, T5EncoderRecipe},
        };
        let device = Default::default();
        let config = T5Config {
            vocab_size: 8,
            d_model: 4,
            d_kv: 2,
            d_ff: 8,
            num_layers: 2,
            num_decoder_layers: Some(1),
            num_heads: 2,
            relative_attention_num_buckets: 8,
            relative_attention_max_distance: 32,
            dropout_rate: 0.0,
            ..T5Config::default()
        };
        Ad::seed(&device, 921);
        let encoder = T5EncoderRecipe::from_config(config)
            .unwrap()
            .init::<Ad>(&device)
            .unwrap();
        let source = || CausalLmInput {
            token_ids: Tensor::from_ints([[1, 2, 3]], &device),
            token_mask: None,
            position_ids: None,
        };
        let weights = || Tensor::<Ad, 3>::from_data([[[1.0, 0.5, -0.5, 2.0]]], &device);
        let direct = encoder
            .encode(source(), &ForwardOptions::default())
            .unwrap()
            .hidden_state;
        let direct_grads = (direct.clone() * weights()).sum().backward();
        let state = encoder.embed(source()).unwrap();
        let state = encoder.run_layers(state, 0..1).unwrap();
        let mut hooks = Hooks::new();
        hooks.edit(Layer::encoder(1).mlp_activation().token(0), |hidden| hidden);
        let state = encoder
            .run_layers_with_hooks(state, 1..2, &mut hooks)
            .unwrap();
        let staged = encoder.normalize(state).unwrap().hidden;
        assert_eq!(direct.into_data(), staged.clone().into_data());
        let staged_grads = (staged * weights()).sum().backward();
        struct Compare<'a> {
            direct: &'a <Ad as burn::tensor::backend::AutodiffBackend>::Gradients,
            staged: &'a <Ad as burn::tensor::backend::AutodiffBackend>::Gradients,
            count: usize,
        }
        impl ModuleVisitor<Ad> for Compare<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<Ad, D>>) {
                let direct = param
                    .val()
                    .grad(self.direct)
                    .expect("direct gradient")
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                let staged = param
                    .val()
                    .grad(self.staged)
                    .expect("staged gradient")
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                for (left, right) in direct.into_iter().zip(staged) {
                    assert!((left - right).abs() < 1e-6);
                }
                self.count += 1;
            }
        }
        let mut compare = Compare {
            direct: &direct_grads,
            staged: &staged_grads,
            count: 0,
        };
        encoder.visit(&mut compare);
        assert!(compare.count > 15);
    }

    #[derive(Module, Debug)]
    struct TiedProjection<B: Backend> {
        weight: Param<Tensor<B, 2>>,
    }

    impl<B: Backend> TiedProjection<B> {
        fn init(device: &B::Device) -> Self {
            Self {
                weight: Param::from_tensor(Tensor::from_data([[0.5, 0.1], [-0.2, 0.4]], device)),
            }
        }
        fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            // One registered owner, with two uses that must accumulate gradients.
            input
                .matmul(self.weight.val())
                .matmul(self.weight.val().transpose())
        }
    }

    #[test]
    fn single_owner_tied_weights_accumulate_both_uses_and_resume_as_one_parameter() {
        let device = Default::default();
        let model = TiedProjection::<Ad>::init(&device);
        let input = || Tensor::from_data([[1.0, 0.0], [0.0, 1.0]], &device);
        let loss = model.forward(input()).square().sum();
        let grads = loss.backward();
        let gradient = model
            .weight
            .val()
            .grad(&grads)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // d ||W W^T||_F^2 / dW = 4 W W^T W.
        let expected = model.weight.val().inner();
        let expected = (expected
            .clone()
            .matmul(expected.clone().transpose())
            .matmul(expected)
            * 4.0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (actual, expected) in gradient.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-7);
        }
        let grads = GradientsParams::from_grads(grads, &model);
        let mut optimizer = SgdConfig::new().init();
        let model = optimizer.step(0.01, model, grads);
        let id = model.weight.id;
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let bytes = recorder.record(model.clone().into_record(), ()).unwrap();
        let restored =
            TiedProjection::<Ad>::init(&device).load_record(recorder.load(bytes, &device).unwrap());
        let mut ids = ParameterIds(Vec::new());
        restored.visit(&mut ids);
        assert_eq!(ids.0, [id]);
        assert_eq!(restored.num_params(), 4);
        assert_eq!(
            restored.forward(input()).into_data(),
            model.forward(input()).into_data()
        );
    }

    struct ParameterIds(Vec<ParamId>);
    impl ModuleVisitor<Ad> for ParameterIds {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<Ad, D>>) {
            self.0.push(param.id);
        }
    }

    #[derive(Module, Debug)]
    struct DuplicateOwners<B: Backend> {
        first: Linear<B>,
        second: Linear<B>,
    }

    #[test]
    fn preflight_rejects_any_trainable_duplicate_and_accepts_frozen_duplicates() {
        let device = Default::default();
        for (first_trainable, second_trainable) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let projection = LinearConfig::new(2, 2).with_bias(false).init::<Ad>(&device);
            let mut model = DuplicateOwners {
                first: projection.clone(),
                second: projection,
            };
            if !first_trainable {
                model.first = model.first.no_grad();
            }
            if !second_trainable {
                model.second = model.second.no_grad();
            }
            assert_eq!(model.first.weight.id, model.second.weight.id);
            let result = validate_trainable_module(&model);
            if first_trainable || second_trainable {
                let message = result.unwrap_err().to_string();
                assert!(message.contains("first.weight"));
                assert!(message.contains("second.weight"));
                assert!(message.contains("Register shared weights once"));
            } else {
                result.unwrap();
            }
            assert_eq!(model.first.weight.val().is_require_grad(), first_trainable);
            assert_eq!(
                model.second.weight.val().is_require_grad(),
                second_trainable
            );
        }
    }

    #[test]
    fn preflight_keeps_unique_parameters_lazy_and_accepts_one_owner_reused_in_forward() {
        let device = Default::default();
        let model = LinearConfig::new(2, 3).init::<Ad>(&device);
        assert!(!model.weight.is_initialized());
        assert!(!model.bias.as_ref().unwrap().is_initialized());
        validate_trainable_module(&model).unwrap();
        assert!(!model.weight.is_initialized());
        assert!(!model.bias.as_ref().unwrap().is_initialized());
        validate_trainable_module(&TiedProjection::<Ad>::init(&device)).unwrap();
    }

    #[derive(Module, Debug)]
    struct ColumnModel<B: Backend> {
        projection: Linear<B>,
    }

    impl<B: Backend> ColumnModel<B> {
        fn init(input_width: usize, output_width: usize, device: &B::Device) -> Self {
            let config = LinearConfig::new(input_width, output_width)
                .with_layout(LinearLayout::Col)
                .with_bias(false);
            Self {
                projection: operations::init_linear(&config, device),
            }
        }

        fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            operations::linear(input, &self.projection).unwrap()
        }
    }

    #[test]
    fn column_records_saved_from_training_owner_reload_evaluate_and_resume_exactly() {
        let device = Default::default();
        for (input_width, output_width) in [(2, 3), (3, 3)] {
            Ad::seed(&device, 991);
            let mut model = ColumnModel::<Ad>::init(input_width, output_width, &device);
            let input = || {
                Tensor::from_data(
                    TensorData::new(
                        (0..4 * input_width)
                            .map(|i| i as f32 * 0.25 - 0.5)
                            .collect::<Vec<_>>(),
                        [4, input_width],
                    ),
                    &device,
                )
            };
            let mut optimizer = AdamConfig::new().init();
            for _ in 0..4 {
                let loss = model.forward(input()).square().mean();
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(0.01, model, grads);
            }

            let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
            // Save the original owner; its mapper writes [output,input].
            let bytes = recorder.record(model.clone().into_record(), ()).unwrap();
            let optimizer_bytes = recorder.record(optimizer.to_record(), ()).unwrap();
            let mut restored = ColumnModel::<Ad>::init(input_width, output_width, &device)
                .load_record(recorder.load(bytes, &device).unwrap());
            let mut restored_optimizer = AdamConfig::new()
                .init()
                .load_record(recorder.load(optimizer_bytes, &device).unwrap());
            assert_eq!(restored.projection.weight.id, model.projection.weight.id);
            assert_eq!(
                restored.projection.weight.val().dims(),
                [input_width, output_width]
            );
            assert_eq!(
                restored.valid().forward(input().inner()).into_data(),
                model.forward(input()).into_data()
            );
            let grads = GradientsParams::from_grads(
                model.forward(input()).square().mean().backward(),
                &model,
            );
            let restored_grads = GradientsParams::from_grads(
                restored.forward(input()).square().mean().backward(),
                &restored,
            );
            model = optimizer.step(0.01, model, grads);
            restored = restored_optimizer.step(0.01, restored, restored_grads);
            assert_eq!(
                restored.projection.weight.val().into_data(),
                model.projection.weight.val().into_data()
            );
        }
    }

    #[test]
    fn converted_evaluation_records_expose_burn_mapper_loss_including_square_weights() {
        let device = Default::default();
        for (input_width, output_width) in [(2, 3), (3, 3)] {
            Ad::seed(&device, 552);
            let model = ColumnModel::<Ad>::init(input_width, output_width, &device);
            let input = Tensor::from_data(
                TensorData::new(
                    (0..input_width).map(|i| i as f32 + 1.0).collect::<Vec<_>>(),
                    [1, input_width],
                ),
                &device,
            );
            let expected = model.forward(input.clone()).into_data();
            // valid() preserves forward values but loses the column save mapper.
            let evaluation = model.valid();
            assert_eq!(
                evaluation.forward(input.clone().inner()).into_data(),
                expected
            );
            let good_record = model.clone().into_record();
            let bad_record = evaluation.into_record();
            assert_eq!(
                good_record.projection.weight.val().dims(),
                [output_width, input_width]
            );
            assert_eq!(
                bad_record.projection.weight.val().dims(),
                [input_width, output_width]
            );
            let bad = ColumnModel::<Cpu>::init(input_width, output_width, &device)
                .load_record(bad_record);
            let actual = operations::linear(input.clone().inner(), &bad.projection);
            if input_width != output_width {
                assert!(
                    actual.is_err(),
                    "nonsquare transpose must fail shape validation"
                );
            } else {
                let actual = actual.unwrap().into_data().to_vec::<f32>().unwrap();
                let expected = expected.to_vec::<f32>().unwrap();
                assert!(
                    actual
                        .into_iter()
                        .zip(expected)
                        .any(|(a, b)| (a - b).abs() > 1e-4),
                    "square transpose can silently alter predictions"
                );
            }

            // Converting back to autodiff does not restore the lost mapper.
            let converted_back = model.valid().train::<Ad>();
            let converted_record = converted_back.into_record();
            assert_eq!(
                converted_record.projection.weight.val().dims(),
                [input_width, output_width]
            );
        }
    }
}
