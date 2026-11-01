//! The x86-64 tier ladder, walked top-down on cached CPUID flags once per matmul
//! call. Nothing here is a build-time decision: the crate compiles for the plain
//! `x86-64` baseline, and the same binary runs tier v4 on a Zen 4 server and
//! tier v1 on an old Core 2. `#[target_feature]` is applied to the kernel
//! functions only, so it never changes codegen anywhere else.

use crate::simd::{scalar, x86_64};

pub(crate) fn dense_q8(out: &mut [f32], w: &[i8], scale: &[f32], m: usize, n: usize, qx: &[i8]) {
    if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512vnni")
    {
        unsafe { x86_64::vnni::dense_q8(out, w, scale, m, n, qx) };
        return;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { x86_64::avx2::dense_q8(out, w, scale, m, n, qx) };
        return;
    }
    if std::arch::is_x86_feature_detected!("sse4.1") {
        unsafe { x86_64::sse41::dense_q8(out, w, scale, m, n, qx) };
        return;
    }
    if std::arch::is_x86_feature_detected!("sse2") {
        unsafe { x86_64::sse2::dense_q8(out, w, scale, m, n, qx) };
        return;
    }
    scalar::dense_q8(out, w, scale, m, n, qx);
}

pub(crate) fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { x86_64::sparse8x4::sparse_sgemv8x4(out, w, idx, rows, x) };
        return;
    }
    crate::simd::scalar::sparse_sgemv8x4(out, w, idx, rows, x);
}

pub(crate) fn dense_sgemv_blocks(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { x86_64::dense_blocks::dense_sgemv_blocks(out, w, m, n, x) };
        return;
    }
    crate::simd::scalar::dense_sgemv_blocks(out, w, m, n, x);
}

pub(crate) fn dense_sgemv(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { x86_64::dense::dense_sgemv(out, w, m, n, x) };
        return;
    }
    crate::simd::scalar::dense_sgemv(out, w, m, n, x);
}

pub(crate) fn xcorr(out: &mut [f32], x: &[f32], y: &[f32], n: usize, max_pitch: usize) {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { x86_64::xcorr::xcorr8(out, x, y, n, max_pitch) };
        return;
    }
    if std::arch::is_x86_feature_detected!("sse2") {
        unsafe { x86_64::xcorr::xcorr4(out, x, y, n, max_pitch) };
        return;
    }
    crate::simd::scalar::xcorr(out, x, y, n, max_pitch);
}

#[cfg(feature = "legacy-model")]
pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    // Tier v4 is not used here: `vpdpbusd` is unsigned × signed, so it needs the
    // per-layer bias correction that only `dense_q8` carries, and legacy rows are
    // 22–64 bytes — too short for a 64-byte kernel to pay for itself.
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { x86_64::avx2::dot(w, x) };
    }
    if std::arch::is_x86_feature_detected!("sse4.1") {
        return unsafe { x86_64::sse41::dot(w, x) };
    }
    if std::arch::is_x86_feature_detected!("sse2") {
        return unsafe { x86_64::sse2::dot(w, x) };
    }
    scalar::i8_dot(w, x)
}
