//! Small, deterministic properties of the public experiment interfaces.
//!
//! Scalar indexing and F64 categorical probabilities are independent oracles.
//! No model checkpoints, accelerator backend, or training loop is required.

#![cfg(feature = "flex")]

use std::cell::Cell;

use burn::{
    backend::Flex,
    module::{Module, Param},
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};
use proptest::prelude::*;
use transformers::{
    Error,
    attention::{AttentionPrecisionMode, AttentionProjection, GroupedQueryAttentionConfig},
    intervention::{Axis, Hooks, Layer, Point, PointValidity},
    training::{cross_entropy, kl_divergence, log_softmax},
    transfer::{TiePolicy, TransferPlan},
};

type Cpu = Flex<f32, i32>;

/// Read a small CPU tensor without changing its rank in the tested graph.
fn values<const D: usize>(tensor: Tensor<Cpu, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

/// Construct a Boolean mask with explicit row-major geometry.
fn boolean<const D: usize>(values: Vec<bool>, shape: [usize; D]) -> Tensor<Cpu, D, Bool> {
    Tensor::<Cpu, D, Int>::from_data(
        TensorData::new(values.into_iter().map(i32::from).collect::<Vec<_>>(), shape),
        &Default::default(),
    )
    .bool()
}

/// Compute categorical log probabilities with a scalar F64 accumulator.
fn reference_log_probabilities(row: &[f32]) -> Vec<f64> {
    let maximum = row
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let normalizer = row
        .iter()
        .map(|&x| (f64::from(x) - maximum).exp())
        .sum::<f64>()
        .ln();
    row.iter()
        .map(|&x| f64::from(x) - maximum - normalizer)
        .collect()
}

#[derive(Module, Debug)]
struct ParameterBlock<B: Backend> {
    weight: Param<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
struct ParameterSet<B: Backend> {
    blocks: Vec<ParameterBlock<B>>,
}

/// Make arbitrary alias groups whose values identify their original owner.
fn parameter_set(groups: &[usize], width: usize, offset: f32) -> ParameterSet<Cpu> {
    let owners = (0..groups.len())
        .map(|index| {
            Param::from_tensor(Tensor::full(
                [2, width],
                offset + index as f32,
                &Default::default(),
            ))
        })
        .collect::<Vec<_>>();
    ParameterSet {
        blocks: groups
            .iter()
            .map(|&group| ParameterBlock {
                weight: owners[group].clone(),
            })
            .collect(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 48,
        rng_seed: proptest::test_runner::RngSeed::Fixed(20260910),
        ..ProptestConfig::default()
    })]

    #[test]
    fn semantic_selection_edits_only_the_named_cartesian_slice(
        batch in 1usize..4, tokens in 1usize..6, channels in 1usize..8,
        permutation in 0usize..6, row in 0usize..20, token in 0usize..20,
        first in 0usize..20, length in 0usize..20, offset in -30i16..30,
    ) {
        let layouts = [
            [Axis::Batch, Axis::Token, Axis::Channel],
            [Axis::Batch, Axis::Channel, Axis::Token],
            [Axis::Token, Axis::Batch, Axis::Channel],
            [Axis::Token, Axis::Channel, Axis::Batch],
            [Axis::Channel, Axis::Batch, Axis::Token],
            [Axis::Channel, Axis::Token, Axis::Batch],
        ];
        let axes = layouts[permutation];
        let dims = axes.map(|axis| match axis {
            Axis::Batch => batch, Axis::Token => tokens, Axis::Channel => channels, _ => unreachable!(),
        });
        let (row, token, first) = (row % batch, token % tokens, first % channels);
        let end = first + 1 + length % (channels - first);
        let original = (0..batch * tokens * channels).map(|i| i as f32 + f32::from(offset)).collect::<Vec<_>>();
        let input = Tensor::<Cpu, 3>::from_data(TensorData::new(original.clone(), dims), &Default::default());
        let point = Point::custom("property.permuted_hidden", axes);
        let selection = point.batch(row).token(token).channels(first..end);
        let mut hooks = Hooks::new();
        let clean = hooks.capture(&selection);
        hooks.edit(&selection, |x| x * 2.0 + 1.0);
        let edited = hooks.capture(&selection);
        let output = hooks.run(vec![point.info()], |hooks| hooks.apply(point.clone(), input.clone())).unwrap();
        let mut expected = original.clone();
        let mut selected = Vec::new();
        for index in 0..original.len() {
            let coordinate = [index / (dims[1] * dims[2]), index / dims[2] % dims[1], index % dims[2]];
            let inside = axes.iter().zip(coordinate).all(|(axis, position)| match axis {
                Axis::Batch => position == row, Axis::Token => position == token,
                Axis::Channel => (first..end).contains(&position), _ => unreachable!(),
            });
            if inside {
                selected.push(original[index]);
                expected[index] = original[index] * 2.0 + 1.0;
            }
        }
        let selected_shape = axes.map(|axis| if axis == Axis::Channel { end - first } else { 1 });
        prop_assert_eq!(hooks.get(&clean).unwrap().dims(), selected_shape);
        prop_assert_eq!(values(hooks.get(&clean).unwrap()), selected.clone());
        prop_assert_eq!(values(hooks.get(&edited).unwrap()), selected.iter().map(|x| x * 2.0 + 1.0).collect::<Vec<_>>());
        prop_assert_eq!(values(output), expected);
        hooks.replace(selection, hooks.get(&clean).unwrap());
        let restored = hooks.run(vec![point.info()], |hooks| hooks.apply(point, input)).unwrap();
        prop_assert_eq!(values(restored), original);
    }

    #[test]
    fn malformed_selections_never_invoke_an_edit_or_publish_partial_captures(
        tokens in 1usize..9, channels in 1usize..9, kind in 0usize..7,
    ) {
        let point = Layer::decoder(0).mlp_activation();
        let selection = match kind {
            0 => point.head(0),
            1 => point.channels(0..channels).channels(0..1),
            2 => point.channels(channels..channels),
            3 => point.channels(channels..0),
            4 => point.token(tokens),
            5 => point.token(usize::MAX),
            _ => point.channels(0..usize::MAX),
        };
        let called = Cell::new(false);
        let mut hooks = Hooks::<Cpu>::new();
        let saved = hooks.capture(&point);
        hooks.edit(selection, |x| { called.set(true); x });
        let result = hooks.run(vec![point.info()], |hooks| {
            hooks.apply(point, Tensor::ones([1, tokens, channels], &Default::default()))
        });
        prop_assert!(result.is_err());
        prop_assert!(!called.get());
        prop_assert!(hooks.get(&saved).is_err());
    }

    #[test]
    fn broadcast_validity_erases_poison_and_survives_ordered_edits(
        dims in (1usize..4, 1usize..5, 1usize..6), broadcast in 0u8..8,
        bits in proptest::collection::vec(any::<bool>(), 60), mode in 0usize..3,
    ) {
        let dims = [dims.0, dims.1, dims.2];
        let shape = std::array::from_fn(|axis| if broadcast & (1 << axis) != 0 { 1 } else { dims[axis] });
        let mask = (0..shape.iter().product()).map(|i| match mode { 0 => false, 1 => true, _ => bits[i] }).collect::<Vec<_>>();
        let valid = (0..dims.iter().product()).map(|index| {
            let coordinate = [index / (dims[1] * dims[2]), index / dims[2] % dims[1], index % dims[2]];
            let coordinate = std::array::from_fn::<_, 3, _>(|axis| if shape[axis] == 1 { 0 } else { coordinate[axis] });
            mask[(coordinate[0] * shape[1] + coordinate[1]) * shape[2] + coordinate[2]]
        }).collect::<Vec<_>>();
        let input = valid.iter().enumerate().map(|(i, valid)| if *valid { i as f32 } else { f32::NAN }).collect::<Vec<_>>();
        let point = Layer::encoder(0).residual_post();
        let validity = PointValidity::new(point.clone(), boolean(mask, shape)).unwrap();
        let mut hooks = Hooks::new();
        let clean = hooks.capture(&point);
        hooks.edit(&point, |x| x + 3.0);
        let edited = hooks.capture(&point);
        let output = hooks.run_with_validity(vec![point.info()], vec![validity], |hooks| {
            hooks.apply(point.clone(), Tensor::from_data(TensorData::new(input, dims), &Default::default()))
        }).unwrap();
        let expected = valid.iter().enumerate().map(|(i, valid)| if *valid { i as f32 + 3.0 } else { 0.0 }).collect::<Vec<_>>();
        prop_assert_eq!(values(output), expected.clone());
        prop_assert_eq!(values(hooks.get(&edited).unwrap()), expected);
        prop_assert!(values(hooks.get(&clean).unwrap()).iter().all(|value| value.is_finite()));
        let unmasked = hooks.run(vec![point.info()], |hooks| hooks.apply(point, Tensor::ones(dims, &Default::default()))).unwrap();
        prop_assert_eq!(values(unmasked), vec![4.0; dims.iter().product()]);
    }

    #[test]
    fn attention_constraints_preserve_exact_exclusions_after_both_edit_layers(
        batch in 1usize..3, heads in 1usize..4, queries in 1usize..5, keys in 1usize..6,
        causal in any::<bool>(), flags in proptest::collection::vec(any::<bool>(), 120),
        alpha in 0u8..5, beta in 0u8..3,
    ) {
        let shape = [batch, heads, queries, keys];
        let size = shape.iter().product::<usize>();
        // Every geometry includes a fully excluded row; other rows vary freely.
        let excluded = (0..size).map(|i| i / keys == 0 || flags[i]).collect::<Vec<_>>();
        let device = Default::default();
        let module = GroupedQueryAttentionConfig::new(heads * 2, heads, 1, 2)
            .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax).init::<Cpu>(&device).unwrap();
        let native_values = (0..batch * keys * 2).map(|i| i as f32 / 8.0).collect::<Vec<_>>();
        let projection = AttentionProjection {
            query: Tensor::zeros([batch, heads, queries, 2], &device),
            key: Tensor::zeros([batch, 1, keys, 2], &device),
            value: Tensor::from_data(TensorData::new(native_values.clone(), [batch, 1, keys, 2]), &device),
            gate: None,
        };
        let points = Layer::decoder(0).self_attention();
        let mut hooks = Hooks::new();
        let structural = hooks.capture(points.pattern());
        hooks.edit(points.pattern(), |pattern| pattern + 0.25);
        let consumer = hooks.capture(points.pattern());
        let context = hooks.capture(points.head_context());
        hooks.run(points.all(), |hooks| {
            module.attend_with_pattern(projection, Some(boolean(excluded.clone(), shape)), None, causal, points, hooks,
                |pattern| Ok(pattern * (f32::from(alpha) / 4.0) + f32::from(beta)))
        }).unwrap();
        let before = values(hooks.get(&structural).unwrap());
        let after = values(hooks.get(&consumer).unwrap());
        let context = values(hooks.get(&context).unwrap());
        for b in 0..batch {
            for h in 0..heads {
                for q in 0..queries {
                    let allowed = (0..keys).map(|k| {
                        let index = ((b * heads + h) * queries + q) * keys + k;
                        !excluded[index] && (!causal || k as isize <= q as isize + keys as isize - queries as isize)
                    }).collect::<Vec<_>>();
                    let count = allowed.iter().filter(|&&yes| yes).count();
                    let mut expected_context = [0.0_f64; 2];
                    for (k, allowed) in allowed.into_iter().enumerate() {
                        let index = ((b * heads + h) * queries + q) * keys + k;
                        let expected_before = if allowed { f64::from(alpha) / 4.0 / count as f64 + f64::from(beta) } else { 0.0 };
                        let expected_after = if allowed { expected_before + 0.25 } else { 0.0 };
                        prop_assert!((f64::from(before[index]) - expected_before).abs() < 2e-6);
                        prop_assert!((f64::from(after[index]) - expected_after).abs() < 2e-6);
                        for channel in 0..2 {
                            expected_context[channel] += expected_after * f64::from(native_values[(b * keys + k) * 2 + channel]);
                        }
                    }
                    for channel in 0..2 {
                        let index = ((b * heads + h) * queries + q) * 2 + channel;
                        prop_assert!((f64::from(context[index]) - expected_context[channel]).abs() < 1e-5);
                    }
                }
            }
        }
    }

    #[test]
    fn transfer_preserves_exact_alias_partitions_or_explicitly_unties(
        groups in proptest::collection::vec(0usize..6, 1..7), width in 1usize..6,
    ) {
        let count = groups.len();
        let groups = groups.into_iter().map(|group| group % count).collect::<Vec<_>>();
        let source = parameter_set(&groups, width, 10.0);
        let target = parameter_set(&(0..count).collect::<Vec<_>>(), width, -10.0);
        for untie in [false, true] {
            let plan = TransferPlan::new().copy_matching();
            let plan = if untie { plan.with_ties(TiePolicy::Untie) } else { plan };
            let (copied, report) = plan.apply(&source, &target).unwrap();
            prop_assert_eq!(report.reused.len(), count);
            for i in 0..count {
                prop_assert_eq!(values(copied.blocks[i].weight.val()), vec![10.0 + groups[i] as f32; 2 * width]);
                prop_assert_ne!(copied.blocks[i].weight.id, source.blocks[i].weight.id);
                for j in 0..count {
                    prop_assert_eq!(copied.blocks[i].weight.id == copied.blocks[j].weight.id, i == j || !untie && groups[i] == groups[j]);
                }
            }
        }
        for i in 0..count {
            prop_assert_eq!(values(target.blocks[i].weight.val()), vec![-10.0 + i as f32; 2 * width]);
            prop_assert_eq!(values(source.blocks[i].weight.val()), vec![10.0 + groups[i] as f32; 2 * width]);
        }
    }

    #[test]
    fn a_failed_transfer_leaves_every_original_parameter_and_alias_untouched(
        groups in proptest::collection::vec(0usize..6, 1..7), width in 1usize..6,
        failed_index in 0usize..20, wrong_shape in any::<bool>(),
    ) {
        let count = groups.len();
        let groups = groups.into_iter().map(|group| group % count).collect::<Vec<_>>();
        let source = parameter_set(&(0..count).collect::<Vec<_>>(), width, 10.0);
        let target = parameter_set(&groups, width, -10.0);
        let before = target.blocks.iter().map(|block| (block.weight.id, values(block.weight.val()))).collect::<Vec<_>>();
        let path = format!("blocks.{}.weight", failed_index % count);
        let result = TransferPlan::new().copy_matching().with_ties(TiePolicy::Untie)
            .transform(&path, &path, "deliberately rejected candidate", move |x: Tensor<Cpu, 2>| {
                if wrong_shape { Ok(Tensor::<Cpu, 2>::zeros([3, width], &x.device())) }
                else { Err(Error::configuration("property.transfer", "deliberate callback failure")) }
            }).apply(&source, &target);
        prop_assert!(result.is_err());
        for (block, (id, contents)) in target.blocks.iter().zip(before) {
            prop_assert_eq!(block.weight.id, id);
            prop_assert_eq!(values(block.weight.val()), contents);
        }
        for (i, block) in source.blocks.iter().enumerate() {
            prop_assert_eq!(values(block.weight.val()), vec![10.0 + i as f32; 2 * width]);
        }
    }

    #[test]
    fn stable_losses_match_scalar_probabilities_and_ignore_row_offsets(
        examples in 1usize..5, classes in 1usize..18,
        inputs in proptest::collection::vec(-128i16..129, 68),
        targets in proptest::collection::vec(0usize..30, 4), shift in -1024i16..1025,
    ) {
        let logits = inputs.into_iter().take(examples * classes).map(|x| f32::from(x) / 8.0).collect::<Vec<_>>();
        let labels = targets.into_iter().take(examples).map(|x| (x % classes) as i32).collect::<Vec<_>>();
        let input = Tensor::<Cpu, 2>::from_data(TensorData::new(logits.clone(), [examples, classes]), &Default::default());
        let label_tensor = Tensor::from_data(TensorData::new(labels.clone(), [examples]), &Default::default());
        let observed = values(log_softmax(input.clone(), 1).unwrap());
        let shifted = values(log_softmax(input.clone() + f32::from(shift), 1).unwrap());
        let mut expected_loss = 0.0;
        for (row, label) in logits.chunks_exact(classes).zip(labels) {
            expected_loss -= reference_log_probabilities(row)[label as usize] / examples as f64;
        }
        let loss = cross_entropy(input.clone(), label_tensor).unwrap().into_scalar();
        prop_assert!((f64::from(loss) - expected_loss).abs() < 4e-6);
        for (row_index, row) in logits.chunks_exact(classes).enumerate() {
            let expected = reference_log_probabilities(row);
            let mut sum = 0.0;
            for column in 0..classes {
                let index = row_index * classes + column;
                prop_assert!((f64::from(observed[index]) - expected[column]).abs() < 4e-6);
                prop_assert!((shifted[index] - observed[index]).abs() < 4e-6);
                sum += f64::from(observed[index]).exp();
            }
            prop_assert!((sum - 1.0).abs() < 2e-6);
        }
        prop_assert_eq!(kl_divergence(input.clone(), input).unwrap().into_scalar(), 0.0);
    }

    #[test]
    fn kl_matches_teacher_to_student_expectation_and_batch_replication(
        classes in 1usize..18,
        student in proptest::collection::vec(-64i16..65, 17),
        teacher in proptest::collection::vec(-64i16..65, 17), copies in 1usize..5,
    ) {
        let student = student.into_iter().take(classes).map(|x| f32::from(x) / 8.0).collect::<Vec<_>>();
        let teacher = teacher.into_iter().take(classes).map(|x| f32::from(x) / 8.0).collect::<Vec<_>>();
        let p = reference_log_probabilities(&student);
        let q = reference_log_probabilities(&teacher);
        let expected = p.iter().zip(&q).map(|(&p, &q)| q.exp() * (q - p)).sum::<f64>();
        let device = Default::default();
        let student = Tensor::<Cpu, 2>::from_data(TensorData::new(student.repeat(copies), [copies, classes]), &device);
        let teacher = Tensor::from_data(TensorData::new(teacher.repeat(copies), [copies, classes]), &device);
        let actual = kl_divergence(student, teacher).unwrap().into_scalar();
        prop_assert!((f64::from(actual) - expected).abs() < 5e-6);
        prop_assert!(actual >= -5e-6);
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn stable_loss_gradients_match_softmax_with_maximum_ties(
        classes in 2usize..12, batch in 1usize..4,
        input in proptest::collection::vec(-16i16..1, 33), target in 0usize..20,
    ) {
        type Ad = burn::backend::Autodiff<Cpu>;
        let mut input = input.into_iter().take(batch * classes).map(|x| f32::from(x) / 4.0).collect::<Vec<_>>();
        for row in input.chunks_exact_mut(classes) {
            row[0] = 0.0;
            row[1] = 0.0;
        }
        let logits = Tensor::<Ad, 2>::from_data(TensorData::new(input.clone(), [batch, classes]), &Default::default()).require_grad();
        let targets = Tensor::<Ad, 1, Int>::from_data(TensorData::new(vec![(target % classes) as i32; batch], [batch]), &Default::default());
        let loss = cross_entropy(logits.clone(), targets).unwrap();
        let gradient = values(logits.grad(&loss.backward()).unwrap());
        for (row_index, row) in input.chunks_exact(classes).enumerate() {
            let probabilities = reference_log_probabilities(row);
            let mut row_sum = 0.0;
            for column in 0..classes {
                let expected = (probabilities[column].exp() - f64::from(column == target % classes)) / batch as f64;
                let actual = f64::from(gradient[row_index * classes + column]);
                prop_assert!((actual - expected).abs() < 3e-7);
                row_sum += actual;
            }
            prop_assert!(row_sum.abs() < 5e-7);
        }
    }
}
