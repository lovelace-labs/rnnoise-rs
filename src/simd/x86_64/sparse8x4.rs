//! Tier v3 (AVX2) `sparse_sgemv8x4`: the eight rows of a block accumulate in one
//! ymm register for the whole row group instead of being reloaded and stored per
//! block. Each lane still accumulates in column order, so the result is
//! bit-identical to [`crate::simd::scalar::sparse_sgemv8x4`].

use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
// SAFETY: requires AVX2, checked at runtime by `select` before calling here. `idx`
// is the upstream sparse layout (`rows` a multiple of 8, every `pos` inside `x`) as
// produced by the blob loader; loads are unaligned, arithmetic exact float per lane.
pub(crate) unsafe fn sparse_sgemv8x4(
    out: &mut [f32],
    w: &[f32],
    idx: &[i32],
    rows: usize,
    x: &[f32],
) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let mut wi = 0usize;
    let mut ii = 0usize;
    let mut g = 0usize;
    while g < rows / 8 {
        let cols = idx[ii] as usize;
        ii += 1;
        let mut acc = _mm256_setzero_ps();
        for _ in 0..cols {
            let pos = idx[ii] as usize;
            ii += 1;
            let p = w.as_ptr().add(wi);
            acc = _mm256_add_ps(
                acc,
                _mm256_mul_ps(_mm256_loadu_ps(p), _mm256_set1_ps(x[pos])),
            );
            acc = _mm256_add_ps(
                acc,
                _mm256_mul_ps(_mm256_loadu_ps(p.add(8)), _mm256_set1_ps(x[pos + 1])),
            );
            acc = _mm256_add_ps(
                acc,
                _mm256_mul_ps(_mm256_loadu_ps(p.add(16)), _mm256_set1_ps(x[pos + 2])),
            );
            acc = _mm256_add_ps(
                acc,
                _mm256_mul_ps(_mm256_loadu_ps(p.add(24)), _mm256_set1_ps(x[pos + 3])),
            );
            wi += 32;
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(g * 8), acc);
        g += 1;
    }
}
