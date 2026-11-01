//! Tier v1 — the `x86-64` baseline (SSE2), which the crate is compiled for and
//! which every x86-64 CPU has.
//!
//! SSE2 has `pmaddwd`, but no instruction widens bytes to words, so the sign mask
//! is built with `cmpgt` and interleaved with `punpcklbw`: one extra µop per
//! product pair compared with tier v2's `pmovsxbw`. Exact int32, so it matches the
//! scalar reference bit for bit.

use core::arch::x86_64::*;

#[inline(always)]
unsafe fn widen(v: __m128i) -> __m128i {
    let sign = _mm_cmpgt_epi8(_mm_setzero_si128(), v);
    _mm_unpacklo_epi8(v, sign)
}

#[inline(always)]
unsafe fn hsum4(acc: __m128i) -> i32 {
    let mut v = acc;
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    _mm_cvtsi128_si32(v)
}

// SAFETY: requires SSE2, which every x86-64 CPU has; `select` only calls this
// after `is_x86_feature_detected!("sse2")`. Loads are unaligned, `w.len() == x.len()`
// is guaranteed by the caller, and all arithmetic is exact int32.
pub(crate) unsafe fn dot(w: &[i8], x: &[i8]) -> i32 {
    let n = w.len();
    let mut acc = _mm_setzero_si128();
    let mut k = 0;
    while k + 8 <= n {
        let a = widen(_mm_loadl_epi64(w.as_ptr().add(k) as *const __m128i));
        let b = widen(_mm_loadl_epi64(x.as_ptr().add(k) as *const __m128i));
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
