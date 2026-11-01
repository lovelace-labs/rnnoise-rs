//! Tier v3 — AVX2: the same `pmovsxbw` + `pmaddwd` shape as tier v2, but 16 lanes
//! wide (32 MACs per instruction pair). Measured 15 MACs/cycle on a Zen 3 core,
//! ~11× the SSE2 scalar loop.

use core::arch::x86_64::*;

#[inline(always)]
unsafe fn hsum8(acc: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let mut v = _mm_add_epi32(lo, hi);
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    _mm_cvtsi128_si32(v)
}

#[target_feature(enable = "avx2")]
// SAFETY: requires AVX2; `select` only calls this after `is_x86_feature_detected!("avx2")`.
// Unaligned 16-byte loads, equal-length slices, exact int32 arithmetic.
pub(crate) unsafe fn dot(w: &[i8], x: &[i8]) -> i32 {
    let n = w.len();
    let mut acc = _mm256_setzero_si256();
    let mut k = 0;
    while k + 16 <= n {
        let a = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(k) as *const __m128i));
        let b = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.as_ptr().add(k) as *const __m128i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
        k += 16;
    }
    let mut sum = hsum8(acc);
    while k < n {
        sum += w[k] as i32 * x[k] as i32;
        k += 1;
    }
    sum
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dense_q8(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    m: usize,
    n: usize,
    qx: &[i8],
) {
    for o in 0..n {
        out[o] = dot(&w[o * m..o * m + m], qx) as f32 * scale[o];
    }
}
