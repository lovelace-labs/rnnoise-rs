//! Runtime `dotprod` detection; the portable reference is the fallback.

use crate::simd::{aarch64, scalar};

pub(crate) fn dense_q8(out: &mut [f32], w: &[i8], scale: &[f32], m: usize, n: usize, qx: &[i8]) {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        scalar::dense_q8(out, w, scale, m, n, qx);
        return;
    }
    unsafe { aarch64::dotprod::dense_q8(out, w, scale, m, n, qx) };
}

pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return scalar::i8_dot(w, x);
    }
    unsafe { aarch64::dotprod::dot(w, x) }
}
