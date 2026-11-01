//! Tier: ARM `dotprod` — the `sdot` int8 dot product (4 MACs per lane). Selected at
//! runtime by [`super::select`]; the rest of the crate is compiled for the plain
//! baseline, so enabling `dotprod` here does not force codegen anywhere else.

use core::arch::aarch64::*;

/// Single `sdot` (signed int8 dot-product, 4 MACs per lane) via inline asm — the
/// `vdotq_s32` intrinsic is still unstable on stable Rust, but the instruction is
/// reachable directly. Requires the `dotprod` target feature.
#[target_feature(enable = "dotprod")]
#[inline]
// SAFETY: the `sdot` instruction needs `dotprod`; only reachable from the
// target_feature functions below, which `select` guards with runtime detection.
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut acc = acc;
    core::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    acc
}

#[cfg(feature = "legacy-model")]
// SAFETY: requires the `dotprod` feature; `select` only calls this after
// `is_aarch64_feature_detected!("dotprod")`. Equal-length slices, exact int32.
// SAFETY: requires the `dotprod` feature, checked at runtime by `select` before
// calling here. Equal-length slices, exact int32 arithmetic.
pub(crate) unsafe fn dot(w: &[i8], x: &[i8]) -> i32 {
    let n = w.len();
    let mut acc = vdupq_n_s32(0);
    let mut k = 0;
    while k + 16 <= n {
        acc = sdot(
            acc,
            vld1q_s8(w.as_ptr().add(k)),
            vld1q_s8(x.as_ptr().add(k)),
        );
        k += 16;
    }
    let mut sum = vaddvq_s32(acc);
    while k < n {
        sum += *w.as_ptr().add(k) as i32 * *x.as_ptr().add(k) as i32;
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
    let chunks = m / 16;
    for o in 0..n {
        let row = w.as_ptr().add(o * m);
        let mut acc = vdupq_n_s32(0);
        for k in 0..chunks {
            acc = sdot(
                acc,
                vld1q_s8(row.add(k * 16)),
                vld1q_s8(qx.as_ptr().add(k * 16)),
            );
        }
        let mut sum = vaddvq_s32(acc);
        for k in chunks * 16..m {
            sum += *row.add(k) as i32 * qx[k] as i32;
        }
        out[o] = sum as f32 * scale[o];
    }
}
