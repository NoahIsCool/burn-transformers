//! Compatibility checks and losses for Burn's ordinary training APIs.
//!
//! Models and inserted modules remain ordinary [`burn::module::Module`] values.
//! Use `Autodiff<B>`, `GradientsParams::from_grads`, and Burn optimizers directly.
//! Freezing a floating-point submodule with `Module::no_grad` freezes its
//! parameters while allowing gradients through its input activations. Burn 0.21
//! does not support autodiff through packed quantized operations. Explicitly
//! dequantize on the inner backend before lifting to autodiff when input
//! gradients are needed; this spends memory and computes in floating point.
//! A packed inference prefix can instead supply detached training features.
//! `capture_with_grad` can supply an intermediate loss; ordinary hook captures
//! are detached. A connected capture does not promise that Burn retains that
//! nonleaf tensor's gradient from a different downstream loss.
//!
//! Burn records restore parameters and parameter identifiers. Reconstruct the
//! same architecture and freezing policy before loading a model record, and
//! restore the optimizer record separately to resume its moments and counters.
//! Store the experiment configuration, data order, and scheduler/RNG state too
//! when the experiment uses them. For tied weights, use one registered parameter
//! reused in the forward graph. Burn 0.21 optimizers do not maintain aliases
//! between duplicate module fields with the same parameter identifier.
//! [`validate_trainable_module`] can detect this before constructing an optimizer.
//!
//! Save records from the original training owner: `model.clone().into_record()`.
//! Load into a freshly initialized owner, then call `valid()` for evaluation.
//! Burn 0.21's `valid()` and `from_inner()` drop parameter save/load mappers.
//! Saving the converted evaluation module can therefore corrupt layouts on
//! reload, including silent transposes of square column-layout projections.
//! These conversions are appropriate for forward evaluation; they do not
//! replace the original owner as the source of future training checkpoints.
//!
//! The small loss functions below use differentiable pairwise reductions. This
//! avoids sequential F32 summation losing probability mass over large
//! vocabularies, including Flex's scalar reduction path. Native SIMD reductions
//! can behave differently. They accept finite F32/F64 logits; explicitly cast lower
//! precision logits to F32 before computing loss. Logits remain on their device;
//! cross-entropy label validation synchronizes minimum/maximum scalars. No
//! implicit dtype promotion, gradient detachment, padding policy, or trainer is applied.

use std::collections::HashMap;

use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    tensor::{
        DType, ElementConversion, Int, Tensor,
        backend::{AutodiffBackend, Backend},
    },
};

use crate::{Error, Result};

/// Checks that each trainable parameter identifier has one registered owner.
///
/// Burn 0.21 optimizers update the first owner of a parameter identifier; they
/// do not maintain aliases between duplicated module fields. Register shared
/// weights once and reuse that parameter in the forward graph. This optional
/// preflight accepts duplicate identifiers when **all** owners are frozen.
/// Run it after applying the experiment's freezing policy and before optimizing.
///
/// The first pass reads identifiers only, keeping ordinary modules lazy. If
/// duplicates exist, their owners may be initialized to inspect trainability:
/// Burn does not expose a public lazy trainability query. The check does not
/// alter the freezing policy, tensors, or optimizer state. It does not audit
/// whether every operation in a custom forward graph supports differentiation.
pub fn validate_trainable_module<B: AutodiffBackend, M: Module<B>>(model: &M) -> Result<()> {
    let mut owners = ParameterOwners::default();
    model.visit(&mut owners);
    owners.groups.retain(|_, group| group.paths.len() > 1);
    if owners.groups.is_empty() {
        return Ok(());
    }
    model.visit(&mut DuplicateTrainability {
        groups: &mut owners.groups,
    });
    let mut invalid = owners
        .groups
        .into_values()
        .filter(|group| group.trainable)
        .map(|group| group.paths)
        .collect::<Vec<_>>();
    invalid.sort();
    if let Some(paths) = invalid.first() {
        return Err(Error::configuration(
            "training.parameters",
            format!(
                "duplicate parameter identifier at {} includes a trainable owner; Burn 0.21 optimizers cannot maintain duplicate module fields. Register shared weights once and reuse that parameter in the forward graph",
                paths.join(", ")
            ),
        ));
    }
    Ok(())
}

#[derive(Default)]
struct ParameterOwners {
    path: Vec<String>,
    groups: HashMap<ParamId, ParameterGroup>,
}

#[derive(Default)]
struct ParameterGroup {
    paths: Vec<String>,
    trainable: bool,
}

impl<B: Backend> ModuleVisitor<B> for ParameterOwners {
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.into());
    }

    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<B, D>>) {
        let path = if self.path.is_empty() {
            "$".into()
        } else {
            self.path.join(".")
        };
        self.groups
            .entry(parameter.id)
            .or_default()
            .paths
            .push(path);
    }
}

struct DuplicateTrainability<'a> {
    groups: &'a mut HashMap<ParamId, ParameterGroup>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for DuplicateTrainability<'_> {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<B, D>>) {
        if let Some(group) = self.groups.get_mut(&parameter.id)
            && !group.trainable
        {
            group.trainable = parameter.val().is_require_grad();
        }
    }
}

/// Computes log probabilities along `dim`, preserving shape, dtype, and gradients.
///
/// Subtracts the row maximum and uses a balanced summation tree for the
/// exponential normalizer. Inputs must have nonempty dimensions and finite
/// F32/F64 values. Nonfinite values follow ordinary floating-point arithmetic.
pub fn log_softmax<B: Backend, const D: usize>(
    logits: Tensor<B, D>,
    dim: usize,
) -> Result<Tensor<B, D>> {
    validate_logits(&logits, "training.logits")?;
    if dim >= D {
        return Err(Error::configuration(
            "training.dim",
            format!("expected an axis below rank {D}, found {dim}"),
        ));
    }
    // The shift is algebraically neutral; detaching only this stabilization
    // constant avoids routing irrelevant gradients through a maximum/tie.
    let maximum = logits.clone().max_dim(dim).detach();
    let shifted = logits - maximum;
    let normalizer = sum_dim(shifted.clone().exp(), dim).log();
    Ok(shifted - normalizer)
}

/// Mean categorical cross entropy for logits `[examples, classes]` and integer labels.
///
/// Each label must be in `0..classes`; validation reads its minimum and maximum
/// and therefore synchronizes the device. For padded batches, select the real
/// examples first, or use [`log_softmax`] to build an explicit masked reduction.
/// Loss is in nats and returned as a one-element tensor connected to the logits.
pub fn cross_entropy<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> Result<Tensor<B, 1>> {
    validate_logits(&logits, "training.logits")?;
    let [examples, classes] = logits.dims();
    if targets.dims() != [examples] {
        return Err(Error::shape(
            "training.targets",
            format!("[{examples}]"),
            format!("{:?}", targets.dims()),
        ));
    }
    if targets.device() != logits.device() {
        return Err(Error::configuration(
            "training.targets.device",
            "targets and logits must use the same device",
        ));
    }
    let minimum = targets.clone().min().into_scalar().elem::<i64>();
    let maximum = targets.clone().max().into_scalar().elem::<i64>();
    if minimum < 0 || maximum as u128 >= classes as u128 {
        return Err(Error::configuration(
            "training.targets",
            format!("expected labels in 0..{classes}, found range {minimum}..={maximum}"),
        ));
    }
    let selected = log_softmax(logits, 1)?
        .gather(1, targets.reshape([examples, 1]))
        .reshape([examples]);
    Ok(-sum_dim(selected, 0) / examples as f64)
}

/// Mean `KL(teacher || student)` from two `[examples, classes]` logit tensors.
///
/// Reduces classes by summation, then averages over examples; loss is in nats.
/// Shapes, devices, and F32/F64 dtypes must match. Both arguments remain
/// differentiable: pass `teacher_logits.detach()` for a fixed teacher target.
/// Small negative values near zero can occur from floating-point roundoff.
pub fn kl_divergence<B: Backend>(
    student_logits: Tensor<B, 2>,
    teacher_logits: Tensor<B, 2>,
) -> Result<Tensor<B, 1>> {
    validate_logits(&student_logits, "training.student_logits")?;
    validate_logits(&teacher_logits, "training.teacher_logits")?;
    if student_logits.dims() != teacher_logits.dims() {
        return Err(Error::shape(
            "training.teacher_logits",
            format!("{:?}", student_logits.dims()),
            format!("{:?}", teacher_logits.dims()),
        ));
    }
    if student_logits.device() != teacher_logits.device()
        || student_logits.dtype() != teacher_logits.dtype()
    {
        return Err(Error::configuration(
            "training.teacher_logits",
            "teacher and student must use the same device and dtype",
        ));
    }
    let [examples, _] = student_logits.dims();
    let student = log_softmax(student_logits, 1)?;
    let teacher = log_softmax(teacher_logits, 1)?;
    let per_example = sum_dim(teacher.clone().exp() * (teacher - student), 1).reshape([examples]);
    Ok(sum_dim(per_example, 0) / examples as f64)
}

fn validate_logits<B: Backend, const D: usize>(logits: &Tensor<B, D>, field: &str) -> Result<()> {
    if D == 0 || logits.dims().contains(&0) {
        return Err(Error::configuration(
            field,
            "all dimensions must be nonempty",
        ));
    }
    if !matches!(logits.dtype(), DType::F32 | DType::F64) {
        return Err(Error::configuration(
            field,
            format!(
                "loss requires F32 or F64 logits; explicitly cast {:?} logits to F32",
                logits.dtype()
            ),
        ));
    }
    Ok(())
}

// Adjacent pairs keep every reduction to two terms on every backend. Flattening
// the other dimensions permits the same implementation at arbitrary input rank.
fn sum_dim<B: Backend, const D: usize>(input: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let mut shape = input.dims();
    shape.swap(dim, D - 1);
    let mut width = shape[D - 1];
    let rows = shape[..D - 1].iter().product::<usize>();
    let mut values = input.swap_dims(dim, D - 1).reshape([rows, width]);
    while width > 1 {
        if !width.is_multiple_of(2) {
            let zero = values.clone().narrow(1, 0, 1).zeros_like();
            values = Tensor::cat(vec![values, zero], 1);
            width += 1;
        }
        width /= 2;
        values = values
            .reshape([rows, width, 2])
            .sum_dim(2)
            .reshape([rows, width]);
    }
    shape[D - 1] = 1;
    values.reshape(shape).swap_dims(dim, D - 1)
}

#[cfg(all(test, feature = "flex"))]
#[path = "training_tests.rs"]
mod tests;
