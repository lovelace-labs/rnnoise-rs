//! The x86-64 tier ladder, walked top-down on cached CPUID flags once per matmul
//! call. Nothing here is a build-time decision: the crate compiles for the plain
//! `x86-64` baseline, and the same binary runs tier v4 on a Zen 4 server and
//! tier v1 on an old Core 2. `#[target_feature]` is applied to the kernel
//! functions only, so it never changes codegen anywhere else.
//!
//! The choice is `simd::tier::resolve(ladder)` — the same top-down order, in one place,
//! plus the CI override: a forced tier is capped to the best one each kernel actually
//! implements (a kernel with no v2 form runs its v1 kernel, so a forced v2 is still
//! meaningful coverage).

use crate::simd::{scalar, x86_64, Tier};

/// int8 dot-product ladder: v4 → v3 → v2 → v1.
pub(crate) const INT8_LADDER: &[Tier] = &[
    Tier::Vnni,
    Tier::Avx2,
    Tier::Sse41,
    Tier::Sse2,
    Tier::Scalar,
];

/// Float kernels that only have an AVX2 form.
pub(crate) const AVX2_ONLY_LADDER: &[Tier] = &[Tier::Avx2, Tier::Scalar];

/// Float kernels with an SSE2 and an AVX2 form.
pub(crate) const FLOAT_LADDER: &[Tier] = &[Tier::Avx2, Tier::Sse2, Tier::Scalar];

// SAFETY: every call below is gated by `resolve`, which only returns a tier the runtime
// feature detection confirmed for the exact ISA that kernel needs. A `#[target_feature]`
// function is unsafe by language rule, so this cannot be expressed safely.

pub(crate) fn dense_q8(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    corr: &[i32],
    m: usize,
    n: usize,
    qx: &[i8],
) {
    // `Tier::is_available` for v4 checks `avx512f` + `avx512bw` + `avx512vnni`: `vpdpbusd`
    // needs the VNNI bit, and the tier also uses `set1_epi8`/`add_epi8` for the +128 bias,
    // which are AVX512BW instructions. In practice every CPU with AVX512VNNI has AVX512BW,
    // but the gate should not depend on that.
    match Tier::resolve(INT8_LADDER) {
        Tier::Vnni => unsafe { x86_64::vnni::dense_q8(out, w, scale, corr, m, n, qx) },
        Tier::Avx2 => unsafe { x86_64::avx2::dense_q8(out, w, scale, corr, m, n, qx) },
        Tier::Sse41 => unsafe { x86_64::sse41::dense_q8(out, w, scale, corr, m, n, qx) },
        Tier::Sse2 => unsafe { x86_64::sse2::dense_q8(out, w, scale, corr, m, n, qx) },
        // Tiers of another architecture, or the portable reference.
        _ => scalar::dense_q8(out, w, scale, corr, m, n, qx),
    }
}

pub(crate) fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    match Tier::resolve(AVX2_ONLY_LADDER) {
        Tier::Avx2 => unsafe { x86_64::sparse8x4::sparse_sgemv8x4(out, w, idx, rows, x) },
        _ => crate::simd::scalar::sparse_sgemv8x4(out, w, idx, rows, x),
    }
}

pub(crate) fn dense_sgemv_blocks(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    match Tier::resolve(AVX2_ONLY_LADDER) {
        Tier::Avx2 => unsafe { x86_64::dense_blocks::dense_sgemv_blocks(out, w, m, n, x) },
        _ => crate::simd::scalar::dense_sgemv_blocks(out, w, m, n, x),
    }
}

pub(crate) fn dense_sgemv(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    match Tier::resolve(AVX2_ONLY_LADDER) {
        Tier::Avx2 => unsafe { x86_64::dense::dense_sgemv(out, w, m, n, x) },
        _ => crate::simd::scalar::dense_sgemv(out, w, m, n, x),
    }
}

pub(crate) fn xcorr(out: &mut [f32], x: &[f32], y: &[f32], n: usize, max_pitch: usize) {
    match Tier::resolve(FLOAT_LADDER) {
        Tier::Avx2 => unsafe { x86_64::xcorr::xcorr8(out, x, y, n, max_pitch) },
        Tier::Sse2 => unsafe { x86_64::xcorr::xcorr4(out, x, y, n, max_pitch) },
        _ => crate::simd::scalar::xcorr(out, x, y, n, max_pitch),
    }
}

#[cfg(feature = "legacy-model")]
pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    // Tier v4 is not used here: `vpdpbusd` is unsigned × signed, so it needs the
    // per-layer bias correction that only `dense_q8` carries, and legacy rows are
    // 22–64 bytes — too short for a 64-byte kernel to pay for itself.
    match Tier::resolve(&[Tier::Avx2, Tier::Sse41, Tier::Sse2, Tier::Scalar]) {
        Tier::Avx2 => unsafe { x86_64::avx2::dot(w, x) },
        Tier::Sse41 => unsafe { x86_64::sse41::dot(w, x) },
        Tier::Sse2 => unsafe { x86_64::sse2::dot(w, x) },
        _ => scalar::i8_dot(w, x),
    }
}
