//! Per-architecture router. The actual tier choice happens in `x86_64::select` /
//! `aarch64::select`, at runtime, from cached CPUID flags.

#[cfg(target_arch = "x86_64")]
pub(crate) fn dense_q8(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    corr: &[i32],
    m: usize,
    n: usize,
    qx: &[i8],
) {
    crate::simd::x86_64::select::dense_q8(out, w, scale, corr, m, n, qx);
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn dense_q8(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    corr: &[i32],
    m: usize,
    n: usize,
    qx: &[i8],
) {
    crate::simd::aarch64::select::dense_q8(out, w, scale, corr, m, n, qx);
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn dense_q8(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    corr: &[i32],
    m: usize,
    n: usize,
    qx: &[i8],
) {
    crate::simd::scalar::dense_q8(out, w, scale, corr, m, n, qx);
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn dense_sgemv(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    crate::simd::x86_64::select::dense_sgemv(out, w, m, n, x);
}

/// Block-major dense GEMV: the same values in a permuted layout, so the tier can keep one
/// accumulator per output group in a register. Only x86-64 has a tier; elsewhere the
/// reference is already the fast path.
#[cfg(target_arch = "x86_64")]
pub(crate) fn dense_sgemv_blocks(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    crate::simd::x86_64::select::dense_sgemv_blocks(out, w, m, n, x);
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn dense_sgemv_blocks(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    crate::simd::scalar::dense_sgemv_blocks(out, w, m, n, x);
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn dense_sgemv(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    crate::simd::scalar::dense_sgemv(out, w, m, n, x);
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    crate::simd::x86_64::select::sparse_sgemv8x4(out, w, idx, rows, x);
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    crate::simd::scalar::sparse_sgemv8x4(out, w, idx, rows, x);
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    crate::simd::scalar::sparse_sgemv8x4(out, w, idx, rows, x);
}

/// Pitch cross-correlation. Only x86-64 has a vector tier; on other targets the
/// portable reference (which compilers vectorize well, since each lag is its own
/// chain) is already the fast path.
#[cfg(target_arch = "x86_64")]
pub(crate) fn xcorr(out: &mut [f32], x: &[f32], y: &[f32], n: usize, max_pitch: usize) {
    crate::simd::x86_64::select::xcorr(out, x, y, n, max_pitch);
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn xcorr(out: &mut [f32], x: &[f32], y: &[f32], n: usize, max_pitch: usize) {
    crate::simd::scalar::xcorr(out, x, y, n, max_pitch);
}

#[cfg(feature = "legacy-model")]
#[cfg(target_arch = "x86_64")]
pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    crate::simd::x86_64::select::i8_dot(w, x)
}

#[cfg(feature = "legacy-model")]
#[cfg(target_arch = "aarch64")]
pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    crate::simd::aarch64::select::i8_dot(w, x)
}

#[cfg(feature = "legacy-model")]
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    crate::simd::scalar::i8_dot(w, x)
}
