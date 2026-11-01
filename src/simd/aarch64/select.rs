//! Runtime `dotprod` detection; the portable reference is the fallback.
//!
//! SAFETY: the `sdot` kernels are only reached after `is_aarch64_feature_detected!`
//! confirms `dotprod`. A `#[target_feature]` function is unsafe by language rule;
//! there is no safe equivalent.

use crate::simd::{aarch64, scalar};

pub(crate) fn dense_q8(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    corr: &[i32],
    m: usize,
    n: usize,
    qx: &[i8],
) {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        scalar::dense_q8(out, w, scale, corr, m, n, qx);
        return;
    }
    unsafe { aarch64::dotprod::dense_q8(out, w, scale, corr, m, n, qx) };
}

pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return scalar::i8_dot(w, x);
    }
    unsafe { aarch64::dotprod::dot(w, x) }
}
