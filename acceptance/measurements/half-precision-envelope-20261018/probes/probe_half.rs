//! Scratch probe of Flex half-precision rounding points (not committed).
#![allow(missing_docs)]
use burn::{backend::Flex, tensor::{DType, Tensor, TensorData, activation, bf16, f16, module::attention, ops::AttentionModuleOptions}};

fn vals(t: Tensor<Flex, 1>) -> Vec<f32> { t.cast(DType::F32).into_data().to_vec::<f32>().unwrap() }
fn all_half(dtype: DType) -> Vec<f32> {
    (0..=u16::MAX).map(|b| match dtype { DType::BF16 => bf16::from_bits(b).to_f32(), _ => f16::from_bits(b).to_f32() }).filter(|v| v.is_finite()).collect()
}
fn round_to(dtype: DType, x: f32) -> f32 { match dtype { DType::BF16 => bf16::from_f32(x).to_f32(), _ => f16::from_f32(x).to_f32() } }

#[test]
fn probe_elementwise() {
    let device = Default::default();
    for dtype in [DType::BF16, DType::F16] {
        let xs = all_half(dtype);
        let n = xs.len();
        let input = Tensor::<Flex, 1>::from_data(TensorData::new(xs.clone(), [n]), &device).cast(dtype);
        for (name, f) in [("sigmoid", activation::sigmoid as fn(Tensor<Flex,1>) -> Tensor<Flex,1>), ("tanh", |t: Tensor<Flex,1>| t.tanh()), ("exp", |t: Tensor<Flex,1>| t.exp()), ("gelu_tanh", activation::gelu_approximate)] {
            let half = vals(f(input.clone()));
            let wide = vals(f(input.clone().cast(DType::F32)).cast(dtype));
            let miss = half.iter().zip(&wide).filter(|(a, b)| a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan())).count();
            eprintln!("{dtype:?} {name}: {miss} of {n} differ from the F32 value rounded once");
        }
    }
}

#[test]
fn probe_binary_and_matmul() {
    let device = Default::default();
    let mut state = 12345u64;
    let mut rnd = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0 };
    for dtype in [DType::BF16, DType::F16] {
        let n = 4096;
        let a: Vec<f32> = (0..n).map(|_| round_to(dtype, rnd() as f32 * 3.0)).collect();
        let b: Vec<f32> = (0..n).map(|_| round_to(dtype, rnd() as f32 * 3.0)).collect();
        let ta = Tensor::<Flex, 1>::from_data(TensorData::new(a.clone(), [n]), &device).cast(dtype);
        let tb = Tensor::<Flex, 1>::from_data(TensorData::new(b.clone(), [n]), &device).cast(dtype);
        for (name, got, want) in [("mul", vals(ta.clone() * tb.clone()), a.iter().zip(&b).map(|(x, y)| round_to(dtype, x * y)).collect::<Vec<_>>()),
                                  ("add", vals(ta.clone() + tb.clone()), a.iter().zip(&b).map(|(x, y)| round_to(dtype, x + y)).collect::<Vec<_>>())] {
            let miss = got.iter().zip(&want).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            eprintln!("{dtype:?} {name}: {miss} of {n} differ from the exact result rounded once");
        }
        for k in [16usize, 256, 4096] {
            let rows = 64;
            let a: Vec<f32> = (0..rows * k).map(|_| round_to(dtype, rnd() as f32)).collect();
            let b: Vec<f32> = (0..k).map(|_| round_to(dtype, rnd() as f32)).collect();
            let ta = Tensor::<Flex, 2>::from_data(TensorData::new(a.clone(), [rows, k]), &device).cast(dtype);
            let tb = Tensor::<Flex, 2>::from_data(TensorData::new(b.clone(), [k, 1]), &device).cast(dtype);
            let got = ta.matmul(tb).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
            let mut worst_ulps = 0.0f64; let mut worst_rel_sum = 0.0f64; let mut exact_hits = 0;
            for r in 0..rows {
                let exact: f64 = (0..k).map(|i| a[r * k + i] as f64 * b[i] as f64).sum();
                let abs_sum: f64 = (0..k).map(|i| (a[r * k + i] as f64 * b[i] as f64).abs()).sum();
                let rounded = round_to(dtype, exact as f32) as f64;
                if got[r] as f64 == rounded { exact_hits += 1; }
                let err = (got[r] as f64 - exact).abs();
                let ulp = { let e = (exact.abs().max(1e-30)).log2().floor(); 2f64.powf(e + 1.0 - if dtype == DType::BF16 { 8.0 } else { 11.0 }) };
                worst_ulps = worst_ulps.max(err / ulp);
                worst_rel_sum = worst_rel_sum.max(err / abs_sum);
            }
            eprintln!("{dtype:?} matmul k={k}: {exact_hits} of {rows} rows equal the exact sum rounded once; worst error {worst_ulps:.2} ulp, {worst_rel_sum:.2e} of sum|ab|");
        }
    }
}

#[test]
fn probe_softmax_and_attention() {
    let device = Default::default();
    let mut state = 777u64;
    let mut rnd = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0 };
    for dtype in [DType::BF16, DType::F16] {
        let (b, h, q, k, d) = (1, 2, 7, 13, 64);
        let mk = |n: usize, s: f64, rnd: &mut dyn FnMut() -> f64| (0..n).map(|_| round_to(dtype, (rnd() * s) as f32)).collect::<Vec<f32>>();
        let qv = mk(b*h*q*d, 1.0, &mut rnd); let kv = mk(b*h*k*d, 1.0, &mut rnd); let vv = mk(b*h*k*d, 1.0, &mut rnd);
        let t = |v: &Vec<f32>, s: usize| Tensor::<Flex, 4>::from_data(TensorData::new(v.clone(), [b, h, s, d]), &device);
        let opts = AttentionModuleOptions { scale: None, softcap: None, is_causal: false };
        let half = attention(t(&qv, q).cast(dtype), t(&kv, k).cast(dtype), t(&vv, k).cast(dtype), None, None, opts.clone());
        eprintln!("{dtype:?} fused attention output dtype {:?}", half.dtype());
        let half = half.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let wide = attention(t(&qv, q), t(&kv, k), t(&vv, k), None, None, opts).into_data().to_vec::<f32>().unwrap();
        let miss = half.iter().zip(&wide).filter(|(x, y)| x.to_bits() != round_to(dtype, **y).to_bits()).count();
        let worst = half.iter().zip(&wide).map(|(x, y)| ((x - y).abs() as f64)).fold(0.0, f64::max);
        eprintln!("{dtype:?} fused attention: {miss} of {} differ from the F32 attention rounded once; worst |diff| {worst:.3e}", half.len());
        let scores = Tensor::<Flex, 2>::from_data(TensorData::new(mk(7*13, 6.0, &mut rnd), [7, 13]), &device).cast(dtype);
        let sm = activation::softmax(scores.clone(), 1).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let sm_wide = activation::softmax(scores.cast(DType::F32), 1).into_data().to_vec::<f32>().unwrap();
        let miss = sm.iter().zip(&sm_wide).filter(|(x, y)| x.to_bits() != round_to(dtype, **y).to_bits()).count();
        eprintln!("{dtype:?} softmax: {miss} of {} differ from the F32 softmax rounded once", sm.len());
    }
}

#[test]
fn probe_sigmoid_exp_mismatches() {
    let device = Default::default();
    let xs = all_half(DType::BF16);
    let n = xs.len();
    let input = Tensor::<Flex, 1>::from_data(TensorData::new(xs.clone(), [n]), &device).cast(DType::BF16);
    for (name, f, g) in [("sigmoid", activation::sigmoid as fn(Tensor<Flex,1>) -> Tensor<Flex,1>, (|x: f64| 1.0 / (1.0 + (-x).exp())) as fn(f64) -> f64), ("exp", |t: Tensor<Flex,1>| t.exp(), (|x: f64| x.exp()) as fn(f64) -> f64)] {
        let half = vals(f(input.clone()));
        let wide = vals(f(input.clone().cast(DType::F32)).cast(DType::BF16));
        let widef32 = vals(f(input.clone().cast(DType::F32)));
        for i in 0..n {
            if half[i].to_bits() != wide[i].to_bits() {
                eprintln!("BF16 {name}({:e}) = {:e} (Burn BF16) vs {:e} (F32 rounded once); F32 value {:e}; exact {:e}; exact rounded {:e}", xs[i], half[i], wide[i], widef32[i], g(xs[i] as f64), round_to(DType::BF16, g(xs[i] as f64) as f32));
            }
        }
    }
}

#[test]
fn probe_casts() {
    let device = Default::default();
    let x = -1.171875e-2f32;
    let t = Tensor::<Flex, 1>::from_floats([x], &device);
    let s = activation::sigmoid(t.clone()).into_data().to_vec::<f32>().unwrap()[0];
    eprintln!("F32 sigmoid bits {:08x} value {:.12e}", s.to_bits(), s);
    let c = Tensor::<Flex, 1>::from_floats([s], &device).cast(DType::BF16).cast(DType::F32).into_data().to_vec::<f32>().unwrap()[0];
    eprintln!("Flex cast {:e}; half {:e}", c, bf16::from_f32(s).to_f32());
    // Sweep: casts of every F32 with a given low half near the tie.
    let mut state = 99u64; let mut bad = 0; let mut total = 0;
    let mut values = Vec::new();
    for hi in 0x3c00u32..0x3c00 + 2000 { for lo in [0x7fffu32, 0x8000, 0x8001, 0x8002, 0x8100, 0x0001, 0xffff] { values.push(f32::from_bits((hi << 16) | lo)); } }
    state = state.wrapping_add(1);
    let n = values.len();
    let flex = Tensor::<Flex, 1>::from_data(TensorData::new(values.clone(), [n]), &device).cast(DType::BF16).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
    let mut by_lo = std::collections::BTreeMap::new();
    for (v, f) in values.iter().zip(&flex) { total += 1; let h = bf16::from_f32(*v).to_f32(); if h.to_bits() != f.to_bits() { bad += 1; *by_lo.entry(v.to_bits() & 0xffff).or_insert(0) += 1; } }
    eprintln!("F32->BF16 cast: {bad} of {total} differ from round-to-nearest-even; by low half: {by_lo:x?}");
    let f16v: Vec<f32> = values.clone();
    let flex16 = Tensor::<Flex, 1>::from_data(TensorData::new(f16v.clone(), [n]), &device).cast(DType::F16).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
    let bad16 = f16v.iter().zip(&flex16).filter(|(v, f)| f16::from_f32(**v).to_f32().to_bits() != f.to_bits()).count();
    eprintln!("F32->F16 cast: {bad16} of {n} differ from round-to-nearest-even");
}

#[test]
fn probe_cast_pattern() {
    let device = Default::default();
    let mut values = Vec::new();
    for hi in [0x3c00u32, 0x3c01, 0x3efe, 0x3eff, 0x4000, 0x4001, 0xbf80, 0xbf81, 0x0080, 0x0081] { for lo in 0u32..=0xffff { values.push(f32::from_bits((hi << 16) | lo)); } }
    let n = values.len();
    let flex = Tensor::<Flex, 1>::from_data(TensorData::new(values.clone(), [n]), &device).cast(DType::BF16).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
    let mut wrong: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for (v, f) in values.iter().zip(&flex) { let h = bf16::from_f32(*v).to_f32(); if h.to_bits() != f.to_bits() { wrong.entry(v.to_bits() >> 16).or_default().push(v.to_bits() & 0xffff); } }
    for (hi, los) in &wrong { eprintln!("upper {hi:04x}: {} wrong, low halves {:04x}..={:04x} (flex rounds {})", los.len(), los.first().unwrap(), los.last().unwrap(), if flex[values.iter().position(|v| v.to_bits() == (hi << 16) | los[0]).unwrap()].to_bits() >> 16 == *hi { "down" } else { "up" }); }
    // A tensor path: does the slice length (SIMD vs scalar tail) matter?
    for len in [1usize, 3, 8, 16, 17, 64] {
        let v = vec![f32::from_bits(0x3efe8001); len];
        let c = Tensor::<Flex, 1>::from_data(TensorData::new(v, [len]), &device).cast(DType::BF16).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        eprintln!("len {len}: {:?}", c.iter().map(|x| x.to_bits() >> 16).collect::<std::collections::BTreeSet<_>>());
    }
}

#[test]
fn probe_scalar_mul() {
    let device = Default::default();
    let scale = (128f64).sqrt().recip();
    for dtype in [DType::BF16, DType::F16] {
        let xs = all_half(dtype).into_iter().filter(|x| x.abs() < 1e4 && x.abs() > 1e-3).collect::<Vec<_>>();
        let n = xs.len();
        let t = Tensor::<Flex, 1>::from_data(TensorData::new(xs.clone(), [n]), &device).cast(dtype);
        let got = vals(t.clone() * scale);
        let got_f32scalar = vals(t.clone().mul_scalar(scale as f32));
        let torch_like: Vec<f32> = xs.iter().map(|x| round_to(dtype, x * scale as f32)).collect();
        let halfscale = round_to(dtype, scale as f32);
        let half_scalar: Vec<f32> = xs.iter().map(|x| round_to(dtype, x * halfscale)).collect();
        let d1 = got.iter().zip(&torch_like).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let d2 = got.iter().zip(&half_scalar).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let d3 = got_f32scalar.iter().zip(&torch_like).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("{dtype:?} x * (1/sqrt(128)): {d1} of {n} differ from round(x * f32 scale) (PyTorch), {d2} from round(x * {dtype:?}(scale)); mul_scalar(f32): {d3}");
    }
}
