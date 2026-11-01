//! Portable reference kernels. These are the definitions the SIMD tiers must match
//! bit for bit, and they are what runs on targets without a detected SIMD path.

pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    let mut acc = 0i32;
    for (&wv, &xv) in w.iter().zip(x) {
        acc += wv as i32 * xv as i32;
    }
    acc
}

/// `out[o] = scale[o] · (w[o] · x)`, one contiguous int8 row per output.
pub(crate) fn dense_q8(out: &mut [f32], w: &[i8], scale: &[f32], m: usize, n: usize, qx: &[i8]) {
    for o in 0..n {
        out[o] = i8_dot(&w[o * m..o * m + m], qx) as f32 * scale[o];
    }
}

/// Port of `sparse_sgemv8x4` (float weights) — the portable reference. Accumulated
/// per column (`out[row] += w[col*8+row] * x[pos+col]`), which is exactly the order
/// upstream `vec.h` uses, so a vectorised version over the 8 rows is bit-identical.
pub(crate) fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let mut wi = 0usize;
    let mut ii = 0usize;
    let mut i = 0usize;
    while i < rows {
        let cols = idx[ii] as usize;
        ii += 1;
        for _ in 0..cols {
            let pos = idx[ii] as usize;
            ii += 1;
            for c in 0..4 {
                for r in 0..8 {
                    out[i + r] += w[wi + c * 8 + r] * x[pos + c];
                }
            }
            wi += 32;
        }
        i += 8;
    }
}

/// Dense float GEMV as a sequence of SAXPYs (j outer, i inner): streams the
/// matrix sequentially (cache-friendly, auto-vectorisable) while preserving the
/// per-`out[i]` accumulation order, so it is bit-identical to the C reference.
pub(crate) fn dense_sgemv(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for (j, &xj) in x[..m].iter().enumerate() {
        let row = &w[j * n..j * n + n];
        for (o, &wv) in out.iter_mut().zip(row.iter()) {
            *o += wv * xj;
        }
    }
}

/// Dense float GEMV over the **block-major** form of the same matrix.
///
/// `w` holds the identical values as [`dense_sgemv`]'s row-major matrix, permuted so
/// that the eight outputs of a group are contiguous over `j`:
/// `w[(o / 8) * 8 * m + j * 8 + (o % 8)]`. Each output still receives its contributions
/// in ascending `j`, so the bits are identical — the layout is a permutation, not an
/// arithmetic change. It exists so a tier can hold one accumulator per group in a
/// register for the whole `j` walk instead of round-tripping the accumulators through
/// memory. Outputs that do not complete a group keep a contiguous run of their own at the
/// end of the buffer.
#[inline]
pub(crate) fn dense_sgemv_blocks(out: &mut [f32], w: &[f32], m: usize, n: usize, x: &[f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let groups = n / 8;
    for g in 0..groups {
        let block = g * 8 * m;
        let mut acc = [0.0f32; 8];
        for (j, &xj) in x[..m].iter().enumerate() {
            for lane in 0..8 {
                acc[lane] += w[block + j * 8 + lane] * xj;
            }
        }
        out[g * 8..g * 8 + 8].copy_from_slice(&acc);
    }
    for o in groups * 8..n {
        let block = groups * 8 * m + (o - groups * 8) * m;
        for (j, &xj) in x[..m].iter().enumerate() {
            out[o] += w[block + j] * xj;
        }
    }
}

/// `xcorr[i] = Σ_{j<n} x[j]·y[i+j]` for `i in 0..max_pitch`.
///
/// Definition: **each lag sums strictly in `j` order**. Four lags are run side by side
/// (one accumulator each) so the loop vectorizes; because each lag keeps its own
/// single accumulator, running four, eight or any other number of lags in parallel
/// gives the identical result — which is what the tiers in
/// [`crate::simd::x86_64::xcorr`] rely on. The leftover lags are done one at a time
/// in the same order.
pub(crate) fn xcorr(out: &mut [f32], x: &[f32], y: &[f32], n: usize, max_pitch: usize) {
    let n4 = n - n % 4;
    let mut i = 0;
    while i + 4 <= max_pitch {
        let (mut c0, mut c1, mut c2, mut c3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for j in 0..n4 {
            let xj = x[j];
            c0 += xj * y[i + j];
            c1 += xj * y[i + 1 + j];
            c2 += xj * y[i + 2 + j];
            c3 += xj * y[i + 3 + j];
        }
        for j in n4..n {
            let xj = x[j];
            c0 += xj * y[i + j];
            c1 += xj * y[i + 1 + j];
            c2 += xj * y[i + 2 + j];
            c3 += xj * y[i + 3 + j];
        }
        out[i] = c0;
        out[i + 1] = c1;
        out[i + 2] = c2;
        out[i + 3] = c3;
        i += 4;
    }
    while i < max_pitch {
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += x[j] * y[i + j];
        }
        out[i] = acc;
        i += 1;
    }
}
