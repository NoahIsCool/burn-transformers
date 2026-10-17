//! Burn 0.21 `attention(.., is_causal: true)` disagrees across kernels when
//! the query and key lengths differ (a cached suffix, chunked prefill).
//!
//! `attention_fallback` (and Flex) align the causal mask bottom-right: query
//! row `i` of `Q` rows is token `i + K - Q` and may attend keys `0..=i + K - Q`.
//! The flash kernels of cubek-attention, which `AttentionStrategy::FlashUnit`,
//! `FlashBlackboxAccelerated` and the autotuner run on the CubeCL backends,
//! mask `row < col`: top-left. The same call therefore returns different
//! values depending on which kernel ran. A square problem is the control:
//! both alignments coincide there.

use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
use burn::backend::{Flex, Metal};
use burn::tensor::{
    Distribution, Tensor, TensorData, TensorPrimitive, backend::Backend, module::attention,
    ops::AttentionModuleOptions,
};
use burn_cubecl::kernel::attention::{AttentionStrategy, attention as cube_attention};

/// The Metal backend without fusion, whose primitives are `CubeTensor`s.
type Raw = CubeBackend<WgpuRuntime, f32, i32, u8>;

const HEADS: usize = 4;
const DIM: usize = 64;

/// Causal attention inputs `[1, HEADS, len, DIM]` with `queries <= keys`.
struct Problem {
    queries: usize,
    keys: usize,
    q: TensorData,
    k: TensorData,
    v: TensorData,
}

impl Problem {
    /// Random inputs for `queries` query rows and `keys` key rows.
    fn new(queries: usize, keys: usize) -> Self {
        let device = Default::default();
        let random = |len: usize| {
            Tensor::<Flex, 4>::random([1, HEADS, len, DIM], Distribution::Default, &device)
                .into_data()
        };
        Self {
            queries,
            keys,
            q: random(queries),
            k: random(keys),
            v: random(keys),
        }
    }

    /// Host F64 reference with query row `i` seeing keys `0..=i + shift`.
    fn reference(&self, shift: usize) -> Vec<f32> {
        let (q, k, v) = (
            self.q.to_vec::<f32>().unwrap(),
            self.k.to_vec::<f32>().unwrap(),
            self.v.to_vec::<f32>().unwrap(),
        );
        let scale = 1.0 / (DIM as f64).sqrt();
        let mut out = vec![0.0f32; HEADS * self.queries * DIM];
        for h in 0..HEADS {
            for i in 0..self.queries {
                let visible = (i + shift + 1).min(self.keys);
                let q_row = &q[(h * self.queries + i) * DIM..][..DIM];
                let scores: Vec<f64> = (0..visible)
                    .map(|j| {
                        let k_row = &k[(h * self.keys + j) * DIM..][..DIM];
                        q_row
                            .iter()
                            .zip(k_row)
                            .map(|(a, b)| *a as f64 * *b as f64)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..DIM {
                    let acc: f64 = (0..visible)
                        .map(|j| weights[j] * v[(h * self.keys + j) * DIM + d] as f64)
                        .sum();
                    out[(h * self.queries + i) * DIM + d] = (acc / total) as f32;
                }
            }
        }
        out
    }
}

/// Causal attention with the default scale.
fn options() -> AttentionModuleOptions {
    AttentionModuleOptions {
        is_causal: true,
        ..Default::default()
    }
}

/// Runs `burn::tensor::module::attention` on backend `B` (autotuned on the GPU backends).
fn run<B: Backend>(problem: &Problem) -> Result<Vec<f32>, String> {
    let device = Default::default();
    let q = Tensor::<B, 4>::from_data(problem.q.clone(), &device);
    let k = Tensor::<B, 4>::from_data(problem.k.clone(), &device);
    let v = Tensor::<B, 4>::from_data(problem.v.clone(), &device);
    // The first call autotunes (and returns a candidate's result); report the second.
    let _ = attention(q.clone(), k.clone(), v.clone(), None, None, options()).into_data();
    Ok(attention(q, k, v, None, None, options())
        .into_data()
        .to_vec()
        .unwrap())
}

/// Runs one explicit CubeCL attention strategy, without autotuning.
fn run_strategy(problem: &Problem, strategy: AttentionStrategy) -> Result<Vec<f32>, String> {
    let device = Default::default();
    let tensor = |data: &TensorData| {
        Tensor::<Raw, 4>::from_data(data.clone(), &device)
            .into_primitive()
            .tensor()
    };
    let out = cube_attention(
        tensor(&problem.q),
        tensor(&problem.k),
        tensor(&problem.v),
        None,
        None,
        options(),
        strategy,
    )
    .map_err(|error| error.to_string().trim().to_string())?;
    Ok(
        Tensor::<Raw, 4>::from_primitive(TensorPrimitive::Float(out))
            .into_data()
            .to_vec()
            .unwrap(),
    )
}

/// Largest absolute difference between two equally long slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Prints every kernel's distance to both causal references for one problem.
fn report(problem: &Problem) {
    let bottom_right = problem.reference(problem.keys - problem.queries);
    let top_left = problem.reference(0);
    println!(
        "causal attention, Q = {} queries over K = {} keys, {HEADS} heads, head_dim {DIM}",
        problem.queries, problem.keys
    );
    println!(
        "{:<30} {:>24} {:>20}",
        "max |out - reference|", "bottom-right (Burn)", "top-left"
    );
    for (name, out) in [
        ("Flex", run::<Flex>(problem)),
        ("Metal, autotuned (default)", run::<Metal>(problem)),
        (
            "Metal, Fallback strategy",
            run_strategy(problem, AttentionStrategy::Fallback),
        ),
        (
            "Metal, FlashUnit strategy",
            run_strategy(problem, AttentionStrategy::FlashUnit),
        ),
    ] {
        match out {
            Ok(out) => println!(
                "{name:<30} {:>24.2e} {:>20.2e}",
                max_abs_diff(&out, &bottom_right),
                max_abs_diff(&out, &top_left)
            ),
            Err(error) => println!("{name:<30} declined: {error}"),
        }
    }
    println!();
}

/// Reports a non-square problem and the square control.
fn main() {
    // A cached suffix / chunked prefill: Q < K.
    report(&Problem::new(512, 4097));
    // Control: Q == K, where both alignments coincide.
    report(&Problem::new(512, 512));
}
