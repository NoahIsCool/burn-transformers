//! gemm 0.19 (gemm-common): on x86-64 with FMA, a small product whose
//! right-hand side is column-major (a transposed view of a row-major matrix)
//! runs the horizontal microkernel, which accumulates each dot product in SIMD
//! lanes and reduces them at the end. The same product with a row-major
//! right-hand side runs the ordinary microkernel, which accumulates along `k`
//! sequentially. The two orders round differently, so `A · B` depends on how
//! `B` is laid out in memory. The route is taken when `m * n <= 256` after
//! gemm's own transposition for a row-major destination (`lhs_cs == 1 &&
//! rhs_rs == 1`); aarch64 has no horizontal kernel and is layout-independent.
//!
//! Burn Flex reaches this through `matmul` with a transposed weight (a tied
//! embedding readout, a `LinearLayout::Col` weight loaded from a checkpoint).

/// Deterministic pseudo-random values in [-1, 1).
fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// `C[m, n] = A[m, k] · B[k, n]`, row-major `C`, with `B` given by its strides.
fn product(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    b_cs: isize,
    b_rs: isize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            k as isize,
            b.as_ptr(),
            b_cs,
            b_rs,
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
    c
}

/// Counts the elements that differ between the two layouts of `B`.
fn main() {
    #[cfg(feature = "patched")]
    gemm_common::gemm::set_layout_independent_accumulation(true);

    println!("arch {}", std::env::consts::ARCH);
    #[cfg(target_arch = "x86_64")]
    println!(
        "fma {}, avx2 {}, avx512f {}",
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("avx512f")
    );
    println!(
        "{:>4} {:>4} {:>5} {:>7} {:>22} {:>14}",
        "m", "n", "k", "m*n", "elements that differ", "max |diff|"
    );
    let mut differing_cases = 0;
    for (m, n, k) in [
        (1, 64, 1024),
        (4, 64, 1024),
        (16, 16, 1024),
        (1, 256, 4096),
        (16, 17, 1024),
        (1, 257, 1024),
        (64, 64, 1024),
    ] {
        let a = values(m * k, 1);
        // Logical B[k, n]; `row_major` holds it as [k, n], `col_major` as its transpose [n, k].
        let row_major = values(k * n, 2);
        let mut col_major = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                col_major[j * k + i] = row_major[i * n + j];
            }
        }
        let contiguous = product(m, n, k, &a, &row_major, 1, n as isize);
        let transposed_view = product(m, n, k, &a, &col_major, k as isize, 1);
        let differ = contiguous
            .iter()
            .zip(&transposed_view)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let max = contiguous
            .iter()
            .zip(&transposed_view)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        differing_cases += (differ > 0) as usize;
        println!(
            "{m:>4} {n:>4} {k:>5} {:>7} {:>13} of {:>6} {max:>14.3e}",
            m * n,
            differ,
            m * n
        );
    }
    println!("{differing_cases} of 7 products depend on the right-hand side's memory layout");
}
