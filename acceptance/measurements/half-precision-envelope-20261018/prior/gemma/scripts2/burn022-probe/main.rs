//! Scratch probe (not committed): does Burn 0.22's Metal backend (wgpu + MSL through CubeCL 0.11)
//! run BF16 tensors end to end?
//!
//! Each step runs under `catch_unwind`, so one failure does not hide the rest, and an F16 control
//! repeats every step, as in the Burn 0.21 probe. Host
//! references are computed in F64 from the already-rounded half-precision inputs, so a deviation
//! measures only the device's arithmetic and output rounding, never input rounding. A kernel whose
//! MSL fails to compile panics on wgpu's device thread, not on the caller's, so such a step can
//! *return* (garbage) instead of panicking: read every step's error figure, not only `PANIC`.
//!
//! Run: `cargo +1.99.1 run --release --offline`.



use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::tensor::{Bool, DType, Device, Int, Tensor, TensorData, activation, bf16, f16};

/// The device under test: Burn 0.22's Metal (wgpu + native MSL) device.
fn default_device() -> Device {
    Device::metal(Default::default())
}

/// Runs one named step, printing its result or the panic message.
fn step(name: &str, body: impl FnOnce() -> String) {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(report) => println!("PROBE {name}: {report}"),
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "non-string panic".to_string());
            println!("PROBE {name}: PANIC {}", message.replace('\n', " | "));
        }
    }
}

/// Deterministic pseudo-random values in `[-scale, scale)` (64-bit LCG).
fn values(count: usize, seed: u64, scale: f64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            (2.0 * unit - 1.0) * scale
        })
        .collect()
}

/// A half-precision element kind with host rounding.
#[derive(Clone, Copy, Debug)]
enum Half {
    /// bfloat16 (8-bit exponent, 8-bit significand).
    Bf16,
    /// IEEE binary16 (5-bit exponent, 11-bit significand).
    F16,
}

impl Half {
    /// Burn's dtype for this kind.
    fn dtype(self) -> DType {
        match self {
            Half::Bf16 => DType::BF16,
            Half::F16 => DType::F16,
        }
    }

    /// Rounds an F64 value to this kind (via F32, exact for the probe's ranges).
    fn round(self, value: f64) -> f64 {
        match self {
            Half::Bf16 => bf16::from_f32(value as f32).to_f64(),
            Half::F16 => f16::from_f32(value as f32).to_f64(),
        }
    }

    /// Unit roundoff (half an ulp of 1).
    fn unit_roundoff(self) -> f64 {
        match self {
            Half::Bf16 => 2f64.powi(-8),
            Half::F16 => 2f64.powi(-11),
        }
    }

    /// Builds host data of this kind from F64 values, returning the rounded values too.
    fn data(self, raw: &[f64], shape: Vec<usize>) -> (TensorData, Vec<f64>) {
        let rounded: Vec<f64> = raw.iter().map(|v| self.round(*v)).collect();
        let data = match self {
            Half::Bf16 => TensorData::new(
                rounded.iter().map(|v| bf16::from_f64(*v)).collect::<Vec<_>>(),
                shape,
            ),
            Half::F16 => TensorData::new(
                rounded.iter().map(|v| f16::from_f64(*v)).collect::<Vec<_>>(),
                shape,
            ),
        };
        (data, rounded)
    }

    /// Uploads F64 values rounded to this kind, returning the tensor and the rounded values.
    fn tensor<const D: usize>(self, raw: &[f64], shape: [usize; D]) -> (Tensor<D>, Vec<f64>) {
        let (data, rounded) = self.data(raw, shape.to_vec());
        (
            Tensor::<D>::from_data(data, (&default_device(), self.dtype())),
            rounded,
        )
    }
}

/// Reads a tensor back as F64, converting on the host from its stored dtype.
fn read<const D: usize>(tensor: Tensor<D>) -> (DType, Vec<f64>) {
    let data = tensor.into_data();
    let dtype = data.dtype();
    let host = match dtype {
        DType::BF16 => data
            .try_to_vec::<bf16>()
            .unwrap()
            .iter()
            .map(|v| v.to_f64())
            .collect(),
        DType::F16 => data
            .try_to_vec::<f16>()
            .unwrap()
            .iter()
            .map(|v| v.to_f64())
            .collect(),
        DType::F32 => data.try_to_vec::<f32>().unwrap().iter().map(|v| *v as f64).collect(),
        other => panic!("unexpected dtype {other:?}"),
    };
    (dtype, host)
}

/// Largest absolute difference, `inf` if any value is non-finite on one side only.
fn max_abs(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| {
            if a.is_nan() && e.is_nan() {
                0.0
            } else if !a.is_finite() || !e.is_finite() {
                if a == e { 0.0 } else { f64::INFINITY }
            } else {
                (a - e).abs()
            }
        })
        .fold(0.0, f64::max)
}

/// Host matrix product in F64 of row-major `[m, k] x [k, n]`, plus `sum |a||b|` per entry.
fn host_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0; m * n];
    let mut scale = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let left = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += left * b[p * n + j];
                scale[i * n + j] += (left * b[p * n + j]).abs();
            }
        }
    }
    (out, scale)
}

/// Matmul check: worst error in units of `u * sum|a||b|` (the standard dot-product bound).
fn matmul_report(kind: Half, m: usize, k: usize, n: usize, seed: u64) -> String {
    let (left, a) = kind.tensor(&values(m * k, seed, 1.0), [m, k]);
    let (right, b) = kind.tensor(&values(k * n, seed + 1, 1.0), [k, n]);
    let (dtype, actual) = read(left.matmul(right));
    let (expected, scale) = host_matmul(&a, &b, m, k, n);
    let u = kind.unit_roundoff();
    let mut worst_rel_scale = 0.0f64;
    let mut nonfinite = 0usize;
    for idx in 0..actual.len() {
        if !actual[idx].is_finite() {
            nonfinite += 1;
            continue;
        }
        let err = (actual[idx] - expected[idx]).abs();
        worst_rel_scale = worst_rel_scale.max(err / (u * scale[idx]));
    }
    format!(
        "out dtype {dtype:?}; max |err| {:.3e}; worst err/(u*sum|a||b|) {worst_rel_scale:.3}; non-finite {nonfinite}",
        max_abs(&actual, &expected),
    )
}

/// Elementwise unary check against the host: worst error in unit roundoffs of the exact result.
fn unary_report(
    kind: Half,
    raw: &[f64],
    op: impl Fn(Tensor<1>) -> Tensor<1>,
    host: impl Fn(f64) -> f64,
) -> String {
    let (tensor, input) = kind.tensor(raw, [raw.len()]);
    let (dtype, actual) = read(op(tensor));
    let expected: Vec<f64> = input.iter().map(|v| host(*v)).collect();
    let u = kind.unit_roundoff();
    let worst = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs() / (u * e.abs().max(1e-30)))
        .fold(0.0, f64::max);
    format!(
        "out dtype {dtype:?}; max |err| {:.3e}; worst err/(u*|exact|) {worst:.2}",
        max_abs(&actual, &expected)
    )
}

/// Probes every capability with the F16 control first, then BF16.


fn main() {
    let device = default_device();
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        step(&format!("declared {dtype:?}"), || {
            format!(
                "supports_dtype={}",
                device.supports_dtype(dtype)
            )
        });
    }

    for kind in [Half::F16, Half::Bf16] {
        let tag = format!("{kind:?}");

        step(&format!("{tag} create+readback"), || {
            let (tensor, rounded) = kind.tensor(&values(1031, 7, 100.0), [1031]);
            let stored = tensor.dtype();
            let (dtype, back) = read(tensor);
            format!(
                "tensor.dtype()={stored:?} readback dtype {dtype:?}; max |diff| {:.3e}",
                max_abs(&back, &rounded)
            )
        });

        step(&format!("{tag} cast from F32 (RNE ties, extremes)"), || {
            let specials: Vec<f32> = vec![
                1.0,
                1.003_906_25, // bf16 tie between 1.0 and 1.0078125 -> 1.0
                1.011_718_75, // bf16 tie -> 1.015625 (even)
                1.000_488_3,  // f16 tie between 1 and 1+2^-10 -> 1.0
                65504.0,
                65520.0, // f16 overflow threshold
                70000.0,
                3.0e38,
                -0.0,
                1.0e-8,
                5.0e-6,
                f32::INFINITY,
                f32::NEG_INFINITY,
                297_984.0,
            ];
            let tensor = Tensor::<1>::from_data(
                TensorData::new(specials.clone(), [specials.len()]),
                (&device, DType::F32),
            );
            let (dtype, back) = read(tensor.cast(kind.dtype()));
            let expected: Vec<f64> = specials.iter().map(|v| kind.round(*v as f64)).collect();
            let mismatches: Vec<String> = back
                .iter()
                .zip(&expected)
                .zip(&specials)
                .filter(|((a, e), _)| !(a == e || (a.is_nan() && e.is_nan())))
                .map(|((a, e), s)| format!("{s:e}->{a:e} (host {e:e})"))
                .collect();
            format!("out dtype {dtype:?}; mismatches vs host RNE: {mismatches:?}")
        });

        step(&format!("{tag} cast to F32 (exact widening)"), || {
            let (tensor, rounded) = kind.tensor(&values(4099, 11, 1.0e4), [4099]);
            let (dtype, back) = read(tensor.cast(DType::F32));
            format!("out dtype {dtype:?}; max |diff| {:.3e}", max_abs(&back, &rounded))
        });

        step(&format!("{tag} cast to the other half type"), || {
            let other = match kind {
                Half::Bf16 => Half::F16,
                Half::F16 => Half::Bf16,
            };
            let (tensor, rounded) = kind.tensor(&values(4099, 12, 100.0), [4099]);
            let expected: Vec<f64> = rounded.iter().map(|v| other.round(*v)).collect();
            let (dtype, back) = read(tensor.cast(other.dtype()));
            format!("out dtype {dtype:?}; max |diff| {:.3e}", max_abs(&back, &expected))
        });

        let raw = values(4096, 13, 4.0);
        step(&format!("{tag} add"), || {
            let (left, a) = kind.tensor(&raw, [raw.len()]);
            let (right, b) = kind.tensor(&values(raw.len(), 17, 4.0), [raw.len()]);
            let (dtype, out) = read(left + right);
            let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| kind.round(x + y)).collect();
            format!(
                "out dtype {dtype:?}; max |err| vs once-rounded sum {:.3e}",
                max_abs(&out, &expected)
            )
        });
        step(&format!("{tag} mul"), || {
            let (left, a) = kind.tensor(&raw, [raw.len()]);
            let (right, b) = kind.tensor(&values(raw.len(), 19, 4.0), [raw.len()]);
            let (dtype, out) = read(left * right);
            let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| kind.round(x * y)).collect();
            format!(
                "out dtype {dtype:?}; max |err| vs once-rounded product {:.3e}",
                max_abs(&out, &expected)
            )
        });
        step(&format!("{tag} exp"), || {
            unary_report(kind, &raw, |t| t.exp(), f64::exp)
        });
        step(&format!("{tag} tanh"), || {
            unary_report(kind, &raw, |t| t.tanh(), f64::tanh)
        });
        step(&format!("{tag} sigmoid"), || {
            unary_report(kind, &raw, activation::sigmoid, |x| 1.0 / (1.0 + (-x).exp()))
        });
        step(&format!("{tag} sqrt"), || {
            let positive: Vec<f64> = raw.iter().map(|v| v.abs() + 1e-3).collect();
            unary_report(kind, &positive, |t| t.sqrt(), f64::sqrt)
        });
        step(&format!("{tag} cos"), || {
            unary_report(kind, &raw, |t| t.cos(), f64::cos)
        });

        step(&format!("{tag} softmax [8,1031] dim 1"), || {
            let (tensor, input) = kind.tensor(&values(8 * 1031, 23, 8.0), [8, 1031]);
            let (dtype, out) = read(activation::softmax(tensor, 1));
            let mut expected = vec![0.0; input.len()];
            for row in 0..8 {
                let slice = &input[row * 1031..(row + 1) * 1031];
                let max = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = slice.iter().map(|v| (v - max).exp()).sum();
                for (j, v) in slice.iter().enumerate() {
                    expected[row * 1031 + j] = (v - max).exp() / sum;
                }
            }
            let worst_row = (0..8)
                .map(|r| (out[r * 1031..(r + 1) * 1031].iter().sum::<f64>() - 1.0).abs())
                .fold(0.0, f64::max);
            format!(
                "out dtype {dtype:?}; max |err| {:.3e} (largest prob {:.3e}); worst |row sum - 1| {worst_row:.3e}",
                max_abs(&out, &expected),
                expected.iter().cloned().fold(0.0, f64::max)
            )
        });

        step(&format!("{tag} reductions sum/max/argmax"), || {
            let (tensor, input) = kind.tensor(&values(4096, 29, 1.0), [4, 1024]);
            let (_, sums) = read(tensor.clone().sum_dim(1));
            let (_, maxes) = read(tensor.clone().max_dim(1));
            let argmax: Vec<i64> = tensor
                .argmax(1)
                .into_data()
                .convert::<i64>()
                .try_to_vec()
                .unwrap();
            let mut detail = Vec::new();
            for r in 0..4 {
                let row = &input[r * 1024..(r + 1) * 1024];
                let sum: f64 = row.iter().sum();
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let arg = row.iter().position(|v| *v == max).unwrap() as i64;
                let abs_sum: f64 = row.iter().map(|v| v.abs()).sum();
                detail.push(format!(
                    "row{r}: sum err/(u*sum|x|) {:.3} max exact {} argmax {}",
                    (sums[r] - sum).abs() / (kind.unit_roundoff() * abs_sum),
                    maxes[r] == max,
                    argmax[r] == arg
                ));
            }
            detail.join("; ")
        });

        step(&format!("{tag} select rows (embedding gather)"), || {
            let (table, rows) = kind.tensor(&values(50 * 64, 53, 1.0), [50, 64]);
            let ids = [2i64, 49, 0, 7, 2];
            let index = Tensor::<1, Int>::from_data(TensorData::new(ids.to_vec(), [5]), &device);
            let (dtype, out) = read(table.select(0, index));
            let expected: Vec<f64> = ids
                .iter()
                .flat_map(|i| rows[*i as usize * 64..(*i as usize + 1) * 64].to_vec())
                .collect();
            format!("out dtype {dtype:?}; max |diff| {:.3e}", max_abs(&out, &expected))
        });

        step(&format!("{tag} swap_dims+reshape copy, cat, mask_fill"), || {
            let (tensor, input) = kind.tensor(&values(2 * 3 * 4, 59, 1.0), [2, 3, 4]);
            let (_, swapped) = read(tensor.clone().swap_dims(1, 2).reshape([2, 12]));
            let mut expected = Vec::new();
            for b in 0..2 {
                for c in 0..4 {
                    for r in 0..3 {
                        expected.push(input[b * 12 + r * 4 + c]);
                    }
                }
            }
            let swap = max_abs(&swapped, &expected);
            let (dtype, joined) = read(Tensor::cat(vec![tensor.clone(), tensor.clone()], 1));
            let cat_ok = joined.len() == 48 && joined[..12] == input[..12] && joined[12..24] == input[..12];
            let mask = Tensor::<3, Bool>::from_data(
                TensorData::new((0..24).map(|i| i % 3 == 0).collect::<Vec<_>>(), [2, 3, 4]),
                &device,
            );
            let (_, filled) = read(tensor.mask_fill(mask, f32::NEG_INFINITY));
            let fill_ok = filled
                .iter()
                .zip(&input)
                .enumerate()
                .all(|(i, (f, x))| if i % 3 == 0 { *f == f64::NEG_INFINITY } else { f == x });
            format!("out dtype {dtype:?}; swap max |diff| {swap:.3e}; cat exact {cat_ok}; mask_fill exact {fill_ok}")
        });

        for (m, k, n) in [(1, 64, 32), (7, 2560, 257), (6, 10240, 256)] {
            step(&format!("{tag} matmul [{m},{k}]x[{k},{n}]"), || {
                matmul_report(kind, m, k, n, 31)
            });
        }

        step(&format!("{tag} batched matmul [2,3,128,96]x[2,3,96,80]"), || {
            let (left, a) = kind.tensor(&values(2 * 3 * 128 * 96, 37, 1.0), [2, 3, 128, 96]);
            let (right, b) = kind.tensor(&values(2 * 3 * 96 * 80, 41, 1.0), [2, 3, 96, 80]);
            let (dtype, out) = read(left.matmul(right));
            let mut worst = 0.0f64;
            for batch in 0..6 {
                let (expected, _) = host_matmul(
                    &a[batch * 128 * 96..(batch + 1) * 128 * 96],
                    &b[batch * 96 * 80..(batch + 1) * 96 * 80],
                    128,
                    96,
                    80,
                );
                worst = worst.max(max_abs(&out[batch * 128 * 80..(batch + 1) * 128 * 80], &expected));
            }
            format!("out dtype {dtype:?}; max |err| {worst:.3e}")
        });
    }
}
