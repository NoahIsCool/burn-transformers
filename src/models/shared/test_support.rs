//! Deterministic fixtures shared by the family-glue unit tests.
//!
//! [`assert_edit_failures_propagate`] and
//! [`assert_staged_edit_failures_propagate`] pin the hook contract that a
//! failing user edit at any emitted point aborts the whole pass, or the staged
//! call, with that point's [`Error::Intervention`]. The modular custom models
//! and the T5, GPT-2, GPT-NeoX, Llama, Gemma, and gpt-oss unit tests use them.
//!
//! [`deterministic_parameters`] gives a randomly initialized model weights
//! that do not depend on Burn's random number generator. Burn Flex keeps one
//! process-wide generator and draws a parameter lazily, on its first read, so
//! a test that seeds it and then builds a model receives weights that depend
//! on which other tests drew in between. That is harmless for tests comparing
//! two paths through the same model, but not for tests whose assertions need
//! particular weights, such as a random drafter that must accept some drafts.

use std::{any::Any, collections::HashMap};

use burn::{
    backend::Flex,
    module::{Module, ModuleMapper, Param, ParamId},
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
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

/// Replaces every non-constant float parameter with values drawn from `seed` alone.
///
/// Each element is uniform on `[-b, b]` with `b = 1 / sqrt(numel / last_dim)`
/// (`1 / sqrt(fan_in)` for Burn's `[d_in, d_out]` linear weights, and `1` for
/// a non-constant vector), from a SplitMix64 stream keyed by `seed`, the
/// parameter's position in visit order, and the element index. Constant
/// parameters (unit norm weights, zero biases) are kept: their values never
/// depend on the generator. A parameter shared between modules (a tied
/// embedding) receives one value. Shapes, dtypes, devices, and trainability
/// are preserved. The distribution is a test fixture, not a model
/// initialization: its only purpose is that the same `seed` gives the same
/// weights however tests are scheduled.
pub(crate) fn deterministic_parameters<B: Backend, M: Module<B>>(module: M, seed: u64) -> M {
    module.map(&mut DeterministicParameters {
        seed,
        next: 0,
        drawn: HashMap::new(),
    })
}

/// [`ModuleMapper`] behind [`deterministic_parameters`].
struct DeterministicParameters {
    /// Seed of every draw.
    seed: u64,
    /// Visit-order position of the next distinct parameter.
    next: u64,
    /// Values already assigned, by parameter, so tied parameters agree.
    drawn: HashMap<ParamId, Box<dyn Any>>,
}

impl<B: Backend> ModuleMapper<B> for DeterministicParameters {
    /// Replaces one parameter, or reuses the value already given to its identity.
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, value, mapper) = param.consume();
        if let Some(tensor) = self
            .drawn
            .get(&id)
            .and_then(|tensor| tensor.downcast_ref::<Tensor<B, D>>())
        {
            return Param::from_mapped_value(id, tensor.clone(), mapper);
        }
        let shape = value.dims();
        let host = value
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("F32 parameter values");
        let replacement = if host.iter().all(|&element| element == host[0]) {
            value
        } else {
            let position = self.next;
            self.next += 1;
            let last = shape[D - 1].max(1);
            let bound = (last as f64 / host.len() as f64).sqrt();
            let drawn = (0..host.len() as u64)
                .map(|element| {
                    let bits = split_mix64(
                        self.seed ^ split_mix64(position) ^ split_mix64(element ^ (1 << 63)),
                    );
                    // The top 24 bits give a uniform fraction of [0, 1) exactly in F32.
                    let unit = (bits >> 40) as f64 / (1u64 << 24) as f64;
                    ((2.0 * unit - 1.0) * bound) as f32
                })
                .collect::<Vec<_>>();
            // `from_data` stores the backend's default float type; cast back.
            Tensor::<B, D>::from_data(TensorData::new(drawn, shape), &value.device())
                .cast(value.dtype())
                .set_require_grad(value.is_require_grad())
        };
        self.drawn.insert(id, Box::new(replacement.clone()));
        Param::from_mapped_value(id, replacement, mapper)
    }
}

/// SplitMix64's output function: a bijective 64-bit mix with full avalanche.
fn split_mix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    //! Checks that [`super::deterministic_parameters`] ignores Burn's generator and keeps structure.

    use burn::{
        module::{Module, Param},
        nn::{Linear, LinearConfig},
        tensor::{DType, Distribution, Tensor, backend::Backend},
    };

    use super::{TestBackend, deterministic_parameters};

    /// A module whose two fields share one parameter identity, as a tied embedding does.
    #[derive(Module, Debug)]
    struct Tied<B: Backend> {
        /// The owning field.
        first: Param<Tensor<B, 2>>,
        /// The tied field: same identity as `first`.
        second: Param<Tensor<B, 2>>,
    }

    /// Reads every element of a parameter as host F32.
    fn host<const D: usize>(param: &Param<Tensor<TestBackend, D>>) -> Vec<f32> {
        param
            .val()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap()
    }

    /// The same seed gives the same weights whatever Burn's generator holds; constants stay.
    #[test]
    fn weights_depend_on_the_seed_alone_and_constants_are_kept() {
        let device = Default::default();
        let build = |burn_seed| {
            TestBackend::seed(&device, burn_seed);
            let linear: Linear<TestBackend> = LinearConfig::new(4, 3).init(&device);
            // Burn's own draw differs with `burn_seed`; the mapped values must not.
            deterministic_parameters(linear, 7)
        };
        let (first, second) = (build(1), build(2));
        assert_eq!(host(&first.weight), host(&second.weight));
        let weights = host(&first.weight);
        let bound = (3.0f32 / 12.0).sqrt();
        assert!(weights.iter().all(|value| value.abs() <= bound));
        assert!(weights.windows(2).any(|pair| pair[0] != pair[1]));
        // Burn initializes the bias uniformly too, so it is replaced as well.
        let bias = first.bias.as_ref().unwrap();
        assert_eq!(host(bias), host(second.bias.as_ref().unwrap()));
        assert_ne!(
            host(&deterministic_parameters(build(1), 8).weight),
            weights,
            "another seed draws other weights"
        );

        let constant = Linear {
            weight: Param::from_tensor(Tensor::<TestBackend, 2>::ones([4, 3], &device)),
            bias: None,
        };
        let kept = deterministic_parameters(constant, 7);
        assert!(host(&kept.weight).iter().all(|&value| value == 1.0));
    }

    /// Tied parameters receive one value; dtype and trainability are preserved.
    #[test]
    fn tied_parameters_agree_and_dtype_and_trainability_are_kept() {
        let device = Default::default();
        let shared = Param::from_tensor(Tensor::<TestBackend, 2>::random(
            [3, 5],
            Distribution::Default,
            &device,
        ));
        let tied = Tied {
            first: shared.clone(),
            second: shared,
        };
        let mapped = deterministic_parameters(tied, 3);
        assert_eq!(mapped.first.id, mapped.second.id);
        assert_eq!(host(&mapped.first), host(&mapped.second));
        assert!(mapped.first.val().is_require_grad() == mapped.second.val().is_require_grad());

        let half = Param::from_tensor(
            Tensor::<TestBackend, 2>::random([2, 4], Distribution::Default, &device)
                .cast(DType::F16),
        )
        .set_require_grad(false);
        let tied = Tied {
            first: half.clone(),
            second: half,
        };
        let mapped = deterministic_parameters(tied, 3);
        assert_eq!(mapped.first.val().dtype(), DType::F16);
        assert!(!mapped.first.val().is_require_grad());
    }
}
