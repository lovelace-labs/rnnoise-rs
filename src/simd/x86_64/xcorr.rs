//! Cross-correlation kernel (the hottest loop of the pitch search), one function
//! per x86-64 ISA tier.
//!
//! `xcorr[i] = Σ_{j<n} x[j]·y[i+j]`, with **one accumulator per lag** and the sum
//! taken strictly in `j` order. Running `lanes` lags in parallel therefore produces
//! exactly the same bits as the portable reference in [`crate::simd::scalar`]: the
//! lanes are independent chains and nothing is reassociated.

use core::arch::x86_64::*;

use crate::simd::scalar;

#[target_feature(enable = "sse2")]
// SAFETY: requires SSE2, which every x86-64 CPU has; `select` only calls this after
// `is_x86_feature_detected!("sse2")`. `y` always holds `n + max_pitch` entries (the
// pitch buffer), so a `lanes`-wide read at lag `i` stays in bounds. Loads and stores
// are unaligned; all arithmetic is plain float multiply and add, never fused.
pub(crate) unsafe fn xcorr4(out: &mut [f32], x: &[f32], y: &[f32], n: usize, max_pitch: usize) {
    let mut i = 0;
    while i + 4 <= max_pitch {
        let mut acc = _mm_setzero_ps();
        for j in 0..n {
            let xv = _mm_set1_ps(x[j]);
            let yv = _mm_loadu_ps(y.as_ptr().add(i + j));
            acc = _mm_add_ps(acc, _mm_mul_ps(xv, yv));
        }
        _mm_storeu_ps(out.as_mut_ptr().add(i), acc);
        i += 4;
    }
    scalar::xcorr(&mut out[i..], x, &y[i..], n, max_pitch - i);
}

#[target_feature(enable = "avx2")]
// SAFETY: requires AVX2; `select` only calls this after runtime detection. Same
// bounds argument as `xcorr4`: eight independent per-lag chains, plain float
// multiply and add (never fused), unaligned loads and stores.
pub(crate) unsafe fn xcorr8(out: &mut [f32], x: &[f32], y: &[f32], n: usize, max_pitch: usize) {
    let mut i = 0;
    while i + 8 <= max_pitch {
        let mut acc = _mm256_setzero_ps();
        for j in 0..n {
            let xv = _mm256_set1_ps(x[j]);
            let yv = _mm256_loadu_ps(y.as_ptr().add(i + j));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(xv, yv));
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(i), acc);
        i += 8;
    }
    scalar::xcorr(&mut out[i..], x, &y[i..], n, max_pitch - i);
}
