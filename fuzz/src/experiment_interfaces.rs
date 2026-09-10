//! Bounded public-interface oracles for activation experiments and training losses.
//!
//! One input chooses one independent check. Physical tensor extents stay at most
//! five, while semantic ranges include machine-integer overflow boundaries.
//! Scalar row-major indexing checks edits and validity; scalar F64 categorical
//! probabilities check losses. Transfer failures must leave both owners intact.
//! The harness never catches panics or treats a successful invalid edit as safe.

use std::cell::Cell;

use burn::{
    backend::Flex,
    module::{Module, Param},
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};
use transformers::{
    Error,
    intervention::{Axis, Hooks, Point, PointValidity},
    training::{cross_entropy, kl_divergence, log_softmax},
    transfer::{TiePolicy, TransferPlan},
};

/// Host-only floating-point backend used by every oracle.
type Cpu = Flex<f32, i32>;

/// Read a bounded tensor for comparison with its independent scalar oracle.
fn values<const D: usize>(tensor: Tensor<Cpu, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

/// Construct a Boolean mask without depending on host Boolean tensor storage.
fn boolean(data: Vec<bool>, shape: [usize; 3]) -> Tensor<Cpu, 3, Bool> {
    Tensor::<Cpu, 3, Int>::from_data(
        TensorData::new(data.into_iter().map(i32::from).collect::<Vec<_>>(), shape),
        &Default::default(),
    )
    .bool()
}

/// Dispatch one bounded semantic-selection, validity, transfer, or loss check.
pub fn exercise(input: &[u8]) {
    let mut bytes = crate::Bytes::new(input);
    match bytes.byte() % 4 {
        0 => selections(&mut bytes),
        1 => validity(&mut bytes),
        2 => transfer(&mut bytes),
        _ => losses(&mut bytes),
    }
}

/// Compare valid slices with scalar coordinates; reject invalid axes and bounds.
fn selections(bytes: &mut crate::Bytes<'_>) {
    let layouts = [
        [Axis::Batch, Axis::Token, Axis::Channel],
        [Axis::Token, Axis::Channel, Axis::Batch],
        [Axis::Batch, Axis::Token, Axis::Token],
        [
            Axis::Other("example"),
            Axis::Other("position"),
            Axis::Channel,
        ],
    ];
    let axes = layouts[usize::from(bytes.byte() % 4)];
    let shape = std::array::from_fn(|_| usize::from(bytes.byte() % 4) + 1);
    let axis = [
        Axis::Batch,
        Axis::Token,
        Axis::Channel,
        Axis::Head,
        Axis::Query,
        Axis::Key,
        Axis::Other("position"),
    ][usize::from(bytes.byte() % 7)];
    let (start, end) = (bytes.size(), bytes.size());
    let locations = axes
        .iter()
        .enumerate()
        .filter_map(|(i, &candidate)| (candidate == axis).then_some(i))
        .collect::<Vec<_>>();
    let accepted = locations.len() == 1 && start < end && end <= shape[locations[0]];
    let point = Point::custom("fuzz.selection", axes);
    let original = (0..shape.iter().product::<usize>())
        .map(|index| index as f32 + 1.0)
        .collect::<Vec<_>>();
    let input = Tensor::<Cpu, 3>::from_data(
        TensorData::new(original.clone(), shape),
        &Default::default(),
    );
    let called = Cell::new(false);
    let mut hooks = Hooks::new();
    let clean = hooks.capture(&point);
    let selected = hooks.capture(point.slice(axis, start..end));
    hooks.edit(point.slice(axis, start..end), |tensor| {
        called.set(true);
        -tensor
    });
    let result = hooks.run(vec![point.info()], |hooks| hooks.apply(point, input));
    assert_eq!(result.is_ok(), accepted);
    assert_eq!(called.get(), accepted);
    if accepted {
        let axis = locations[0];
        let stride = shape[axis + 1..].iter().product::<usize>();
        let mut expected = original.clone();
        let mut expected_selection = Vec::new();
        for (index, item) in expected.iter_mut().enumerate() {
            if (start..end).contains(&(index / stride % shape[axis])) {
                expected_selection.push(original[index]);
                *item = -original[index];
            }
        }
        let mut expected_shape = shape;
        expected_shape[axis] = end - start;
        assert_eq!(hooks.get(&selected).unwrap().dims(), expected_shape);
        assert_eq!(values(hooks.get(&selected).unwrap()), expected_selection);
        assert_eq!(values(hooks.get(&clean).unwrap()), original);
        assert_eq!(values(result.unwrap()), expected);
    } else {
        assert!(hooks.get(&clean).is_err());
        assert!(hooks.get(&selected).is_err());
    }
}

/// Broadcast masks by scalar indexing and prove scope cleanup after either result.
fn validity(bytes: &mut crate::Bytes<'_>) {
    let shape = std::array::from_fn(|_| usize::from(bytes.byte() % 4) + 1);
    let mask_shape = shape.map(|extent| match bytes.byte() % 3 {
        0 => 1,
        1 => extent,
        _ => extent + 1,
    });
    let mask = (0..mask_shape.iter().product::<usize>())
        .map(|_| bytes.byte() & 1 != 0)
        .collect::<Vec<_>>();
    let accepted = mask_shape
        .iter()
        .zip(shape)
        .all(|(&mask, extent)| mask == 1 || mask == extent);
    let valid = (0..shape.iter().product::<usize>())
        .map(|index| {
            let coordinates = [
                index / (shape[1] * shape[2]),
                index / shape[2] % shape[1],
                index % shape[2],
            ];
            let m = std::array::from_fn::<_, 3, _>(|axis| {
                if mask_shape[axis] == 1 {
                    0
                } else {
                    coordinates[axis]
                }
            });
            mask[(m[0] * mask_shape[1] + m[1]) * mask_shape[2] + m[2]]
        })
        .collect::<Vec<_>>();
    let point = Point::custom("fuzz.validity", [Axis::Batch, Axis::Token, Axis::Channel]);
    let validity = PointValidity::new(point.clone(), boolean(mask, mask_shape)).unwrap();
    let called = Cell::new(false);
    let mut hooks = Hooks::new();
    let captured = hooks.capture(&point);
    hooks.edit(&point, |tensor| {
        called.set(true);
        tensor + 5.0
    });
    let original = valid
        .iter()
        .map(|&valid| if valid { 2.0 } else { f32::NAN })
        .collect::<Vec<_>>();
    let result = hooks.run_with_validity(vec![point.info()], vec![validity], |hooks| {
        hooks.apply(
            point.clone(),
            Tensor::from_data(TensorData::new(original, shape), &Default::default()),
        )
    });
    assert_eq!(result.is_ok(), accepted);
    assert_eq!(called.get(), accepted);
    if accepted {
        assert_eq!(
            values(result.unwrap()),
            valid
                .iter()
                .map(|&valid| if valid { 7.0 } else { 0.0 })
                .collect::<Vec<_>>()
        );
        assert!(
            values(hooks.get(&captured).unwrap())
                .iter()
                .all(|x| x.is_finite())
        );
    } else {
        assert!(hooks.get(&captured).is_err());
    }
    let fresh = hooks
        .run(vec![point.info()], |hooks| {
            hooks.apply(point, Tensor::ones(shape, &Default::default()))
        })
        .unwrap();
    assert_eq!(values(fresh), vec![6.0; shape.iter().product()]);
}

/// One registered tensor, so vector indices have unambiguous transfer paths.
#[derive(Module, Debug)]
struct Weight<B: Backend> {
    /// Matrix participating in a generated alias group.
    value: Param<Tensor<B, 2>>,
}

/// Small parameter owner with independently generated alias partitions.
#[derive(Module, Debug)]
struct Owner<B: Backend> {
    /// At most four registered tensors, each with at most eight elements.
    weights: Vec<Weight<B>>,
}

/// Construct consistent aliases and distinguish every group's numerical value.
fn owner(groups: &[usize], width: usize, offset: f32) -> Owner<Cpu> {
    let owners = (0..groups.len())
        .map(|index| {
            Param::from_tensor(Tensor::full(
                [2, width],
                offset + index as f32,
                &Default::default(),
            ))
        })
        .collect::<Vec<_>>();
    Owner {
        weights: groups
            .iter()
            .map(|&group| Weight {
                value: owners[group].clone(),
            })
            .collect(),
    }
}

/// Preserve alias equivalence on success and both original owners on failure.
fn transfer(bytes: &mut crate::Bytes<'_>) {
    let count = usize::from(bytes.byte() % 4) + 1;
    let width = usize::from(bytes.byte() % 4) + 1;
    let mode = bytes.byte() % 5;
    let groups = (0..count)
        .map(|_| usize::from(bytes.byte()) % count)
        .collect::<Vec<_>>();
    let source = owner(&groups, width, 10.0);
    let target = owner(&(0..count).collect::<Vec<_>>(), width, -10.0);
    let source_before = source
        .weights
        .iter()
        .map(|w| (w.value.id, values(w.value.val())))
        .collect::<Vec<_>>();
    let target_before = target
        .weights
        .iter()
        .map(|w| (w.value.id, values(w.value.val())))
        .collect::<Vec<_>>();
    let path = format!("weights.{}.value", usize::from(bytes.byte()) % count);
    let called = Cell::new(false);
    let mut plan = TransferPlan::new().copy_matching();
    if mode != 0 {
        plan = plan.with_ties(TiePolicy::Untie);
    }
    if mode >= 2 {
        plan = plan.transform(&path, &path, "fuzz candidate", |tensor: Tensor<Cpu, 2>| {
            called.set(true);
            match mode {
                2 => Ok(Tensor::<Cpu, 2>::zeros([2, width + 1], &tensor.device())),
                3 => Err(Error::configuration(
                    "fuzz.transfer",
                    "intentional candidate failure",
                )),
                _ => Ok(tensor),
            }
        });
    }
    if mode == 4 {
        plan = plan.copy("nonexistent.value", "weights.0.value");
    }
    let result = plan.apply(&source, &target);
    assert_eq!(result.is_ok(), mode < 2);
    if let Ok((actual, report)) = result {
        assert_eq!(report.reused.len(), count);
        for i in 0..count {
            assert_eq!(values(actual.weights[i].value.val()), source_before[i].1);
            assert_ne!(actual.weights[i].value.id, source_before[i].0);
            for j in 0..count {
                assert_eq!(
                    actual.weights[i].value.id == actual.weights[j].value.id,
                    i == j || mode == 0 && groups[i] == groups[j]
                );
            }
        }
    } else {
        assert_eq!(called.get(), mode != 4);
    }
    for (tensor, (id, data)) in source.weights.iter().zip(source_before) {
        assert_eq!(tensor.value.id, id);
        assert_eq!(values(tensor.value.val()), data);
    }
    for (tensor, (id, data)) in target.weights.iter().zip(target_before) {
        assert_eq!(tensor.value.id, id);
        assert_eq!(values(tensor.value.val()), data);
    }
}

/// Compare losses with scalar probabilities and reject invalid axes and labels.
fn losses(bytes: &mut crate::Bytes<'_>) {
    let examples = usize::from(bytes.byte() % 4) + 1;
    let classes = usize::from(bytes.byte() % 5) + 1;
    let axis = bytes.size();
    let labels = (0..examples)
        .map(|_| i32::from(bytes.byte() % 8) - 1)
        .collect::<Vec<_>>();
    let data = (0..examples * classes)
        .map(|_| (f32::from(bytes.byte()) - 128.0) / 8.0)
        .collect::<Vec<_>>();
    let logits = Tensor::<Cpu, 2>::from_data(
        TensorData::new(data.clone(), [examples, classes]),
        &Default::default(),
    );
    let probabilities = log_softmax(logits.clone(), axis);
    assert_eq!(probabilities.is_ok(), axis < 2);
    if let Ok(logs) = probabilities {
        let actual = values(logs);
        for b in 0..examples {
            for c in 0..classes {
                let line = if axis == 0 {
                    (0..examples)
                        .map(|row| data[row * classes + c])
                        .collect::<Vec<_>>()
                } else {
                    data[b * classes..(b + 1) * classes].to_vec()
                };
                // Values are bounded to [-16,16], so the independent F64 sum
                // needs no stabilization and cannot overflow or underflow.
                let normalizer = line.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln();
                let expected = f64::from(data[b * classes + c]) - normalizer;
                assert!((f64::from(actual[b * classes + c]) - expected).abs() < 4e-6);
            }
        }
    }
    let loss = cross_entropy(
        logits.clone(),
        Tensor::from_data(
            TensorData::new(labels.clone(), [examples]),
            &Default::default(),
        ),
    );
    let accepted = labels
        .iter()
        .all(|&label| label >= 0 && (label as usize) < classes);
    assert_eq!(loss.is_ok(), accepted);
    if let Ok(loss) = loss {
        let mut expected = 0.0;
        for (row, label) in data.chunks_exact(classes).zip(labels) {
            expected += (row.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln()
                - f64::from(row[label as usize]))
                / examples as f64;
        }
        assert!((f64::from(loss.into_scalar()) - expected).abs() < 4e-6);
    }
    assert_eq!(
        kl_divergence(logits.clone(), logits.clone())
            .unwrap()
            .into_scalar(),
        0.0
    );
    assert!(
        kl_divergence(
            logits,
            Tensor::zeros([examples, classes + 1], &Default::default())
        )
        .is_err()
    );
}

#[cfg(test)]
#[path = "experiment_interfaces_tests.rs"]
mod tests;
