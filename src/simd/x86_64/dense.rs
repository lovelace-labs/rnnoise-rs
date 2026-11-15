//! Tier v3 (AVX2) dense float GEMV.
//!
//! One accumulator per output element, kept in a ymm register for eight outputs at a
//! time, with the input streamed in `j` order. That is the same accumulation order as
//! the portable reference in [`crate::simd::scalar`], so the result is bit-identical —
//! the win is purely the wider load/multiply path (the model is bandwidth-bound on the
//! weights, and 8 lanes pull 32 bytes per instruction instead of 16).

use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
// SAFETY: requires AVX2; `select` only calls this after `is_x86_feature_detected!("avx2")`.
// `w` holds `m * n` floats and `x` holds `m`, so every access stays in bounds. Loads are
// unaligned; arithmetic is plain float per lane (never fused), so the per-output
// accumulation order matches the reference.
pub(crate) unsafe fn dense_sgemv(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for (j, &xj) in x[..m].iter().enumerate() {
        let row = &w[j * n..j * n + n];
        let xb = _mm256_set1_ps(xj);
        let mut i = 0;
        while i + 8 <= n {
            let acc = _mm256_loadu_ps(out.as_ptr().add(i));
            let wv = _mm256_loadu_ps(row.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_add_ps(acc, _mm256_mul_ps(wv, xb)),
            );
            i += 8;
        }
        for o in i..n {
            out[o] += row[o] * xj;
        }
    }
}
