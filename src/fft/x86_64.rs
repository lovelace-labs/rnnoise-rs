//! Vector tier (SSE2, the `x86-64` baseline) for the butterfly stages.
//!
//! The vector lanes are independent `j` values of the inner loop, and each element is
//! produced by exactly the same sequence of multiplies and adds as the portable
//! reference in [`crate::fft::scalar`] — so the tier is bit-identical. SSE2 cannot
//! contract a multiply and an add into FMA, which is what keeps the per-element
//! expression order; stages whose `m` is not a multiple of the vector width fall back
//! to the reference.
//!
//! The buffer is AoS (`Cpx`), so each group of four lanes is read and written through
//! two `__m128i` loads with a fixed shuffle; the arithmetic itself runs in split
//! real/imaginary vectors.

use core::arch::x86_64::*;

use crate::fft::{plan::Stage, scalar, spectrum::Cpx};

/// Four `Cpx` values as a real vector and an imaginary vector.
type Pairs = (__m128, __m128);

// `_MM_SHUFFLE(z, y, x, w) = z<<6 | y<<4 | x<<2 | w`, and `_mm_shuffle_ps(a, b, imm)`
// yields `{a[w], a[x], b[y], b[z]}`. Spelled out as constants because the macro is not
// const-stable on the crate's MSRV.
const SHUF_RE: i32 = 136; // _MM_SHUFFLE(2, 0, 2, 0) -> {a0,a2,b0,b2} = re of 4 Cpx
const SHUF_IM: i32 = 221; // _MM_SHUFFLE(3, 1, 3, 1) -> {a1,a3,b1,b3} = im of 4 Cpx

#[inline(always)]
unsafe fn load(p: *const Cpx) -> Pairs {
    let a0 = _mm_castsi128_ps(_mm_loadu_si128(p as *const __m128i));
    let a1 = _mm_castsi128_ps(_mm_loadu_si128(p.add(2) as *const __m128i));
    (
        _mm_shuffle_ps(a0, a1, SHUF_RE),
        _mm_shuffle_ps(a0, a1, SHUF_IM),
    )
}

#[inline(always)]
unsafe fn store(p: *mut Cpx, re: __m128, im: __m128) {
    // Write back as AoS: {re0,im0,re1,im1} then {re2,im2,re3,im3}.
    _mm_storeu_si128(p as *mut __m128i, _mm_castps_si128(_mm_unpacklo_ps(re, im)));
    _mm_storeu_si128(
        p.add(2) as *mut __m128i,
        _mm_castps_si128(_mm_unpackhi_ps(re, im)),
    );
}

#[inline(always)]
unsafe fn cmul(a: Pairs, b: Pairs) -> Pairs {
    (
        _mm_sub_ps(_mm_mul_ps(a.0, b.0), _mm_mul_ps(a.1, b.1)),
        _mm_add_ps(_mm_mul_ps(a.0, b.1), _mm_mul_ps(a.1, b.0)),
    )
}

#[inline(always)]
unsafe fn cadd(a: Pairs, b: Pairs) -> Pairs {
    (_mm_add_ps(a.0, b.0), _mm_add_ps(a.1, b.1))
}

#[inline(always)]
unsafe fn csub(a: Pairs, b: Pairs) -> Pairs {
    (_mm_sub_ps(a.0, b.0), _mm_sub_ps(a.1, b.1))
}

#[inline(always)]
unsafe fn tw(st: &Stage, k: usize, j0: usize) -> Pairs {
    (
        _mm_loadu_ps(st.tw_re.as_ptr().add(k * st.m + j0)),
        _mm_loadu_ps(st.tw_im.as_ptr().add(k * st.m + j0)),
    )
}

// SAFETY: requires SSE2, which every x86-64 CPU has; `dispatch` only calls this after
// runtime detection. Each stage covers the whole `nfft`-element buffer exactly once, so
// the unaligned four-element loads and stores stay in bounds. Arithmetic is plain float
// per lane, never fused.
// SAFETY: requires SSE2, which every x86-64 CPU has; `dispatch` only calls this after
// runtime detection. Each stage covers the whole `nfft`-element buffer exactly once, so
// the unaligned four-element loads and stores stay in bounds. Arithmetic is plain float
// per lane, never fused.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn process(stages: &[Stage], f: &mut [Cpx]) {
    for st in stages {
        if st.m % 4 != 0 {
            scalar::stage(st, f);
            continue;
        }
        match st.p {
            4 => unsafe { bfly4(st, f) },
            3 => unsafe { bfly3(st, f) },
            5 => unsafe { bfly5(st, f) },
            2 => scalar::stage(st, f),
            p => unreachable!("unsupported radix {p}"),
        }
    }
}

unsafe fn bfly4(st: &Stage, f: &mut [Cpx]) {
    let m = st.m;
    let m2 = 2 * m;
    let m3 = 3 * m;
    for i in 0..st.n {
        for j0 in (0..m).step_by(4) {
            let p = f.as_mut_ptr().add(i * st.mm + j0);
            let a = load(p);
            let s0 = cmul(load(p.add(m)), tw(st, 1, j0));
            let s1 = cmul(load(p.add(m2)), tw(st, 2, j0));
            let s2 = cmul(load(p.add(m3)), tw(st, 3, j0));

            let s5 = csub(a, s1);
            let cur = cadd(a, s1);
            let s3 = cadd(s0, s2);
            let s4 = csub(s0, s2);

            store(p.add(m2), _mm_sub_ps(cur.0, s3.0), _mm_sub_ps(cur.1, s3.1));
            store(p, _mm_add_ps(cur.0, s3.0), _mm_add_ps(cur.1, s3.1));
            store(p.add(m), _mm_add_ps(s5.0, s4.1), _mm_sub_ps(s5.1, s4.0));
            store(p.add(m3), _mm_sub_ps(s5.0, s4.1), _mm_add_ps(s5.1, s4.0));
        }
    }
}

unsafe fn bfly3(st: &Stage, f: &mut [Cpx]) {
    let m = st.m;
    let m2 = 2 * m;
    let epi3 = _mm_set1_ps(st.epi3_i);
    let half = _mm_set1_ps(0.5);
    for i in 0..st.n {
        for j0 in (0..m).step_by(4) {
            let p = f.as_mut_ptr().add(i * st.mm + j0);
            let a = load(p);
            let s1 = cmul(load(p.add(m)), tw(st, 1, j0));
            let s2 = cmul(load(p.add(m2)), tw(st, 2, j0));
            let s3 = cadd(s1, s2);
            let s0 = csub(s1, s2);
            let s0 = (_mm_mul_ps(s0.0, epi3), _mm_mul_ps(s0.1, epi3));

            // `f[idx + m] = f[idx] - 0.5 * s3` first, exactly as in C.
            let b = csub(a, (_mm_mul_ps(s3.0, half), _mm_mul_ps(s3.1, half)));
            store(p, _mm_add_ps(a.0, s3.0), _mm_add_ps(a.1, s3.1));
            store(p.add(m2), _mm_add_ps(b.0, s0.1), _mm_sub_ps(b.1, s0.0));
            store(p.add(m), _mm_sub_ps(b.0, s0.1), _mm_add_ps(b.1, s0.0));
        }
    }
}

unsafe fn bfly5(st: &Stage, f: &mut [Cpx]) {
    let m = st.m;
    let m2 = 2 * m;
    let m3 = 3 * m;
    let m4 = 4 * m;
    let ya_r = _mm_set1_ps(st.ya.r);
    let ya_i = _mm_set1_ps(st.ya.i);
    let yb_r = _mm_set1_ps(st.yb.r);
    let yb_i = _mm_set1_ps(st.yb.i);
    for i in 0..st.n {
        for j0 in (0..m).step_by(4) {
            let p = f.as_mut_ptr().add(i * st.mm + j0);
            let s0 = load(p);
            let s1 = cmul(load(p.add(m)), tw(st, 1, j0));
            let s2 = cmul(load(p.add(m2)), tw(st, 2, j0));
            let s3 = cmul(load(p.add(m3)), tw(st, 3, j0));
            let s4 = cmul(load(p.add(m4)), tw(st, 4, j0));

            let s7 = cadd(s1, s4);
            let s10 = csub(s1, s4);
            let s8 = cadd(s2, s3);
            let s9 = csub(s2, s3);

            store(
                p,
                _mm_add_ps(s0.0, _mm_add_ps(s7.0, s8.0)),
                _mm_add_ps(s0.1, _mm_add_ps(s7.1, s8.1)),
            );

            // `ya`/`yb` enter `s5`/`s11` through their real parts only, and
            // `s6`/`s12` through their imaginary parts — the C code is explicit about
            // which part feeds which lane, so the constants are broadcast per use.
            let s5 = (
                _mm_add_ps(
                    s0.0,
                    _mm_add_ps(_mm_mul_ps(s7.0, ya_r), _mm_mul_ps(s8.0, yb_r)),
                ),
                _mm_add_ps(
                    s0.1,
                    _mm_add_ps(_mm_mul_ps(s7.1, ya_r), _mm_mul_ps(s8.1, yb_r)),
                ),
            );
            let s6 = (
                _mm_add_ps(_mm_mul_ps(s10.1, ya_i), _mm_mul_ps(s9.1, yb_i)),
                _mm_sub_ps(
                    _mm_setzero_ps(),
                    _mm_add_ps(_mm_mul_ps(s10.0, ya_i), _mm_mul_ps(s9.0, yb_i)),
                ),
            );
            store(p.add(m), _mm_sub_ps(s5.0, s6.0), _mm_sub_ps(s5.1, s6.1));
            store(p.add(m4), _mm_add_ps(s5.0, s6.0), _mm_add_ps(s5.1, s6.1));

            let s11 = (
                _mm_add_ps(
                    s0.0,
                    _mm_add_ps(_mm_mul_ps(s7.0, yb_r), _mm_mul_ps(s8.0, ya_r)),
                ),
                _mm_add_ps(
                    s0.1,
                    _mm_add_ps(_mm_mul_ps(s7.1, yb_r), _mm_mul_ps(s8.1, ya_r)),
                ),
            );
            let s12 = (
                _mm_sub_ps(_mm_mul_ps(s9.1, ya_i), _mm_mul_ps(s10.1, yb_i)),
                _mm_sub_ps(_mm_mul_ps(s10.0, yb_i), _mm_mul_ps(s9.0, ya_i)),
            );
            store(
                p.add(m2),
                _mm_add_ps(s11.0, s12.0),
                _mm_add_ps(s11.1, s12.1),
            );
            store(
                p.add(m3),
                _mm_sub_ps(s11.0, s12.0),
                _mm_sub_ps(s11.1, s12.1),
            );
        }
    }
}
