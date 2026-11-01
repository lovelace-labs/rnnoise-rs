//! Tier v2 — `x86-64-v2` (SSE4.2). `pmovsxbw` is the SSE4.1 half we actually use:
//! it widens eight int8 to int16 in one µop, replacing tier v1's `cmpgt` +
//! `punpcklbw` sign trick. Same exact int32 result.

use core::arch::x86_64::*;

#[inline(always)]
unsafe fn hsum4(acc: __m128i) -> i32 {
    let mut v = acc;
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    _mm_cvtsi128_si32(v)
}

#[target_feature(enable = "sse4.1")]
// SAFETY: requires SSSE3/SSE4.1 (`pmovsxbw`); `select` only calls this after the
// runtime detection. Unaligned loads, equal-length slices, exact int32 arithmetic.
pub(crate) unsafe fn dot(w: &[i8], x: &[i8]) -> i32 {
    let n = w.len();
    let mut acc = _mm_setzero_si128();
    let mut k = 0;
    while k + 8 <= n {
        let a = _mm_cvtepi8_epi16(_mm_loadl_epi64(w.as_ptr().add(k) as *const __m128i));
        let b = _mm_cvtepi8_epi16(_mm_loadl_epi64(x.as_ptr().add(k) as *const __m128i));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(a, b));
        k += 8;
    }
    let mut sum = hsum4(acc);
    while k < n {
        sum += w[k] as i32 * x[k] as i32;
        k += 1;
    }
    sum
}

#[target_feature(enable = "sse4.1")]
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
