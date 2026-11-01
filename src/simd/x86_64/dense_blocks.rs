//! Tier v3 (AVX2) dense float GEMV over the block-major layout.
//!
//! Four groups of eight outputs are accumulated in four `ymm` registers for the whole `j`
//! walk, so the accumulators never round-trip through memory — that is the whole point of
//! the layout, and it also puts four independent add chains in flight. Every output still
//! receives its contributions in ascending `j`, so this is bit-identical to
//! [`crate::simd::scalar::dense_sgemv_blocks`] (and hence to the row-major reference on the
//! same values). Measured on a Zen 3 core: 384×1152 goes from ~20.7 to ~15.3 µs per call.

use core::arch::x86_64::*;

/// Groups accumulated in parallel, one `ymm` register each.
const GROUPS_PER_STEP: usize = 4;

#[target_feature(enable = "avx2")]
// SAFETY: requires AVX2, checked at runtime by `select` before calling here. `w` holds
// `m * n` floats and `x` holds `m`, so every access stays in bounds. Loads are unaligned;
// arithmetic is plain float per lane (never fused), so the per-output accumulation order
// matches the reference.
pub(crate) unsafe fn dense_sgemv_blocks(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let groups = n / 8;
    let mut g = 0;
    while g + GROUPS_PER_STEP <= groups {
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let b0 = w.as_ptr().add(g * 8 * m);
        let b1 = b0.add(8 * m);
        let b2 = b1.add(8 * m);
        let b3 = b2.add(8 * m);
        for (j, &xj) in x[..m].iter().enumerate() {
            let xb = _mm256_set1_ps(xj);
            let off = j * 8;
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(b0.add(off)), xb));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(b1.add(off)), xb));
            a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(b2.add(off)), xb));
            a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(b3.add(off)), xb));
        }
        let o = out.as_mut_ptr().add(g * 8);
        _mm256_storeu_ps(o, a0);
        _mm256_storeu_ps(o.add(8), a1);
        _mm256_storeu_ps(o.add(16), a2);
        _mm256_storeu_ps(o.add(24), a3);
        g += GROUPS_PER_STEP;
    }

    // Groups left over: one accumulator, still register-resident.
    while g < groups {
        let mut a = _mm256_setzero_ps();
        let b = w.as_ptr().add(g * 8 * m);
        for (j, &xj) in x[..m].iter().enumerate() {
            a = _mm256_add_ps(
                a,
                _mm256_mul_ps(_mm256_loadu_ps(b.add(j * 8)), _mm256_set1_ps(xj)),
            );
        }
        _mm256_storeu_ps(out.as_mut_ptr().add(g * 8), a);
        g += 1;
    }

    // Outputs that do not complete a group keep a run of their own at the end, exactly as
    // the reference defines it.
    for o in groups * 8..n {
        let block = groups * 8 * m + (o - groups * 8) * m;
        for (j, &xj) in x[..m].iter().enumerate() {
            out[o] += w[block + j] * xj;
        }
    }
}
