//! Runtime `dotprod` detection; the portable reference is the fallback.
//!
//! The choice is `simd::tier::resolve(ladder)`, so CI can force the `dotprod` tier on a
//! runner that has it (and only there).
//!
//! SAFETY: the `sdot` kernels are only reached after `resolve` confirms `dotprod` through
//! `is_aarch64_feature_detected!`. A `#[target_feature]` function is unsafe by language
//! rule; there is no safe equivalent.

use crate::simd::{aarch64, scalar, Tier};

pub(crate) const LADDER: &[Tier] = &[Tier::Dotprod, Tier::Scalar];

pub(crate) fn dense_q8(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    corr: &[i32],
    m: usize,
    n: usize,
    qx: &[i8],
) {
    // `sdot` is signed × signed, so there is no bias correction here — `corr` is the
    // zero vector for this architecture and the tier is exact by construction.
    match Tier::resolve(LADDER) {
        Tier::Dotprod => unsafe { aarch64::dotprod::dense_q8(out, w, scale, corr, m, n, qx) },
        _ => scalar::dense_q8(out, w, scale, corr, m, n, qx),
    }
}

#[cfg(feature = "legacy-model")]
pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    match Tier::resolve(LADDER) {
        Tier::Dotprod => unsafe { aarch64::dotprod::dot(w, x) },
        _ => scalar::i8_dot(w, x),
    }
}
