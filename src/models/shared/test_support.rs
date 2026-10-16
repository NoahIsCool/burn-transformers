//! Deterministic fixtures shared by the family-glue unit tests.
//!
//! [`assert_edit_failures_propagate`] and
//! [`assert_staged_edit_failures_propagate`] pin the hook contract that a
//! failing user edit at any emitted point aborts the whole pass, or the staged
//! call, with that point's [`Error::Intervention`]. The modular custom models
//! and the T5, GPT-2, GPT-NeoX, Llama, Gemma, and gpt-oss unit tests use them.

use burn::{
    backend::Flex,
    tensor::{Bool, Int, Tensor, backend::Backend},
};

use crate::{
    CausalLmInput, Error, Result,
    architecture::{StagedModel, TokenMask},
    intervention::{Hooks, Point, PointInfo},
};

/// CPU Flex backend with F32 activations and portable I32 indices.
pub(super) type TestBackend = Flex<f32, i32>;

/// Asserts that two errors have the same variant and the same rendered message.
pub(super) fn assert_same_error(actual: Error, expected: Error) {
    assert_eq!(
        std::mem::discriminant(&actual),
        std::mem::discriminant(&expected),
        "{actual} != {expected}"
    );
    assert_eq!(actual.to_string(), expected.to_string());
}

/// Builds a `[rows, columns]` integer tensor on the default device.
pub(super) fn ints<const R: usize, const C: usize>(
    values: [[i64; C]; R],
) -> Tensor<TestBackend, 2, Int> {
    Tensor::from_data(values, &Default::default())
}

/// Builds a token mask (`true == valid`) from host booleans.
pub(super) fn mask<const R: usize, const C: usize>(
    values: [[bool; C]; R],
) -> TokenMask<TestBackend> {
    TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_data(
        values,
        &Default::default(),
    ))
}

/// Builds a token mask from a row-major host vector.
pub(super) fn mask_from_rows(rows: &[Vec<bool>]) -> TokenMask<TestBackend> {
    let batch = rows.len();
    let sequence = rows.first().map_or(0, Vec::len);
    let values = rows.iter().flatten().copied().collect::<Vec<_>>();
    TokenMask::new(crate::architecture::bool_tensor_from_values(
        values,
        [batch, sequence],
        &Default::default(),
    ))
}

/// Reads a float tensor into row-major host values.
pub(super) fn floats<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("f32 tensor data")
}

/// Reads an integer tensor into row-major host values.
pub(super) fn int_values<const D: usize>(tensor: Tensor<TestBackend, D, Int>) -> Vec<i64> {
    tensor.into_data().iter::<i64>().collect()
}

/// Cause returned by every edit that [`register_failing_edit`] installs.
const EDIT_FAILURE: &str = "deliberate edit failure";

/// Registers an edit at `point` whose callback always returns an error.
///
/// The point's runtime rank selects the typed [`Point`]; activation points of
/// this crate have ranks one to five (`[batch, token, hidden]` residuals,
/// `[batch, head, query, key]` attention planes, and recurrent states).
///
/// # Panics
///
/// Panics for a rank outside `1..=5`, which no family emits.
pub(crate) fn register_failing_edit<B: Backend>(hooks: &mut Hooks<'_, B>, point: &PointInfo) {
    /// Installs the failing edit at a point of known rank `D`.
    fn typed<B: Backend, const D: usize>(hooks: &mut Hooks<'_, B>, point: &PointInfo) {
        let point = Point::<D>::try_from(point.clone()).expect("catalog point has this rank");
        hooks.try_edit(point, |_| {
            Err(Error::BackendCapability(EDIT_FAILURE.into()))
        });
    }
    match point.rank() {
        1 => typed::<B, 1>(hooks, point),
        2 => typed::<B, 2>(hooks, point),
        3 => typed::<B, 3>(hooks, point),
        4 => typed::<B, 4>(hooks, point),
        5 => typed::<B, 5>(hooks, point),
        rank => panic!("no activation point has rank {rank}: {}", point.name),
    }
}

/// Asserts that a failing edit at each of `points` aborts `run` at that point.
///
/// For every point, a fresh [`Hooks`] holds exactly one edit whose callback
/// fails; `run` must then return [`Error::Intervention`] naming that point,
/// with the callback's error as its cause. Points whose edits the hook
/// machinery rejects by design (derived sink probabilities) must still fail
/// at that point. Running every catalog point proves that each family's
/// emission sites propagate user failures instead of dropping or remapping
/// them, whichever block, branch, or stack emitted the point.
pub(crate) fn assert_edit_failures_propagate<B: Backend, T>(
    points: &[PointInfo],
    mut run: impl FnMut(&mut Hooks<'_, B>) -> Result<T>,
) {
    assert!(!points.is_empty(), "the catalog lists no points");
    for point in points {
        let mut hooks = Hooks::<B>::new();
        register_failing_edit(&mut hooks, point);
        match run(&mut hooks) {
            Err(Error::Intervention {
                point: failed,
                reason,
                ..
            }) => {
                assert_eq!(failed, point.name, "{reason}");
                assert!(
                    reason.contains(EDIT_FAILURE) || point.name.ends_with("sink_probability"),
                    "{}: {reason}",
                    point.name
                );
            }
            Err(other) => panic!(
                "{}: expected an intervention error, found {other}",
                point.name
            ),
            Ok(_) => panic!("{}: a failing edit did not abort the pass", point.name),
        }
    }
}

/// Asserts that every staged call propagates a failing edit at each point it emits.
///
/// `points` is the model's complete catalog for one stack: its embedding
/// point, the points of every block, and its final-norm point. The embedding
/// point must fail [`StagedModel::embed_with_hooks`], each block point must
/// fail [`StagedModel::run_layers_with_hooks`] over all blocks, and the
/// final-norm point must fail [`StagedModel::normalize_with_hooks`].
pub(crate) fn assert_staged_edit_failures_propagate<B: Backend, M: StagedModel<B>>(
    model: &M,
    input: &CausalLmInput<B>,
    points: &[PointInfo],
) {
    let stack = model.stack();
    let embedding = stack.embedding().info();
    let final_norm = stack.final_norm().info();
    assert!(points.contains(&embedding) && points.contains(&final_norm));
    let blocks = points
        .iter()
        .filter(|point| **point != embedding && **point != final_norm)
        .cloned()
        .collect::<Vec<_>>();
    let layers = 0..model.num_layers();
    assert_edit_failures_propagate(&[embedding], |hooks| {
        model.embed_with_hooks(input.clone(), hooks)
    });
    let embedded = model.embed(input.clone()).expect("unhooked embedding");
    assert_edit_failures_propagate(&blocks, |hooks| {
        model.run_layers_with_hooks(embedded.clone(), layers.clone(), hooks)
    });
    let ran = model.run_layers(embedded, layers).expect("unhooked blocks");
    assert_edit_failures_propagate(&[final_norm], |hooks| {
        model.normalize_with_hooks(ran.clone(), hooks)
    });
}
