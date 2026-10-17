//! Scheduling-independent random weights for tests outside the library crate.
//!
//! Burn Flex keeps one process-wide random number generator. `Backend::seed`
//! stores it, and every random draw, including each parameter's default
//! initialization, which Burn defers to the parameter's first read, takes the
//! generator, draws, and puts it back. The tests of one binary run on parallel
//! threads, so a test that seeds and then builds a model receives weights that
//! depend on which other tests drew in between, and always on Burn's
//! generator and initializers themselves. [`deterministic_parameters`]
//! replaces every non-constant float parameter with values computed from a
//! seed alone. It is the construction of the library's crate-private test
//! helper of the same name, which integration and example tests cannot reach,
//! so a test whose assertions need particular weights (a logit gap, a margin)
//! receives the same weights on every run and every host.
//!
//! The example tests of `activation_patching` include this file through a
//! `#[path]` attribute; its unit tests run with them.

use std::{any::Any, collections::HashMap};

use burn::{
    module::{Module, ModuleMapper, Param, ParamId},
    tensor::{Tensor, TensorData, backend::Backend},
};

/// Replaces every non-constant float parameter with values drawn from `seed` alone.
///
/// Each element is uniform on `[-b, b]` with `b = 1 / sqrt(numel / last_dim)`
/// (`1 / sqrt(fan_in)` for Burn's `[d_in, d_out]` linear weights, and `1` for
/// a non-constant vector), from a SplitMix64 stream keyed by `seed`, the
/// parameter's position in visit order, and the element index. Constant
/// parameters (unit norm weights, zero biases) are kept: their values never
/// depend on the generator. A parameter shared between modules (a tied
/// embedding) receives one value. Shapes, dtypes, devices, and trainability
/// are preserved. Reading each parameter materializes Burn's own lazy draw
/// first; that value is discarded, so it does not matter which generator
/// state it came from. The distribution is a test fixture, not a model
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
        backend::Flex,
        module::{Module, Param},
        nn::{Linear, LinearConfig},
        tensor::{DType, Distribution, Tensor, backend::Backend},
    };

    use super::deterministic_parameters;

    /// CPU Flex backend with F32 activations and I32 indices.
    type Cpu = Flex<f32, i32>;

    /// A module whose two fields share one parameter identity, as a tied embedding does.
    #[derive(Module, Debug)]
    struct Tied<B: Backend> {
        /// The owning field.
        first: Param<Tensor<B, 2>>,
        /// The tied field: same identity as `first`.
        second: Param<Tensor<B, 2>>,
    }

    /// Reads every element of a parameter as host F32.
    fn host<const D: usize>(param: &Param<Tensor<Cpu, D>>) -> Vec<f32> {
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
            Cpu::seed(&device, burn_seed);
            let linear: Linear<Cpu> = LinearConfig::new(4, 3).init(&device);
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
            weight: Param::from_tensor(Tensor::<Cpu, 2>::ones([4, 3], &device)),
            bias: None,
        };
        let kept = deterministic_parameters(constant, 7);
        assert!(host(&kept.weight).iter().all(|&value| value == 1.0));
    }

    /// Tied parameters receive one value; dtype and trainability are preserved.
    #[test]
    fn tied_parameters_agree_and_dtype_and_trainability_are_kept() {
        let device = Default::default();
        let shared = Param::from_tensor(Tensor::<Cpu, 2>::random(
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

        let half = Param::from_tensor(
            Tensor::<Cpu, 2>::random([2, 4], Distribution::Default, &device).cast(DType::F16),
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

    /// The stream is pinned: these values are the library helper's for the same seed and shape.
    ///
    /// Integration and example tests choose seeds by measured margins, so a
    /// change to the construction would silently move every chosen weight.
    #[test]
    fn the_stream_is_pinned() {
        let device = Default::default();
        let weight = Param::from_tensor(Tensor::<Cpu, 2>::random(
            [2, 2],
            Distribution::Default,
            &device,
        ));
        let mapped = deterministic_parameters(Linear { weight, bias: None }, 7);
        assert_eq!(host(&mapped.weight), PINNED);
    }

    /// Values of a `[2, 2]` weight for seed 7, recorded from the library helper.
    ///
    /// Each literal is the shortest decimal that round-trips to the recorded
    /// F32, so the comparison is exact.
    const PINNED: [f32; 4] = [0.15318744, -0.5272512, 0.50852245, 0.25228176];
}
