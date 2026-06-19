//! LPC analysis + inner-product kernels, ported from `celt_lpc.c` / `pitch.h`
//! (float configuration, so the fixed-point shift macros are all identities).

/// `sum_{i<n} x[i] * y[i]`, using four independent accumulators so the loop
/// vectorizes (the four partials are summed at the end). This reorders the
/// float additions relative to a strict left-to-right sum, which is the hot
/// kernel of the pitch search; the divergence from the C reference is sub-LSB.
#[inline]
pub(crate) fn inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let n4 = n - n % 4;
    for (xc, yc) in x[..n4].chunks_exact(4).zip(y[..n4].chunks_exact(4)) {
        s0 += xc[0] * yc[0];
        s1 += xc[1] * yc[1];
        s2 += xc[2] * yc[2];
        s3 += xc[3] * yc[3];
    }
    let mut xy = (s0 + s1) + (s2 + s3);
    for i in n4..n {
        xy += x[i] * y[i];
    }
    xy
}

/// `dual_inner_prod`: two correlations sharing the `x` operand (vectorized).
#[inline]
pub(crate) fn dual_inner_prod(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    let (mut a0, mut a1, mut b0, mut b1) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let n2 = n - n % 2;
    let mut i = 0;
    while i < n2 {
        a0 += x[i] * y01[i];
        a1 += x[i + 1] * y01[i + 1];
        b0 += x[i] * y02[i];
        b1 += x[i + 1] * y02[i + 1];
        i += 2;
    }
    let (mut a, mut b) = (a0 + a1, b0 + b1);
    while i < n {
        a += x[i] * y01[i];
        b += x[i] * y02[i];
        i += 1;
    }
    (a, b)
}

/// `rnn_pitch_xcorr`: `xcorr[i] = sum_{j<len} x[j] * y[i+j]` for `i in 0..max_pitch`.
///
/// Computes four lags at once, advancing through `x` while keeping a small window
/// of `y` in registers — this reuses each `x` load four times and lets the inner
/// loop vectorize, which matters because this is the hottest kernel of the pitch
/// search. Each `xcorr[i]` still accumulates in `j` order.
pub(crate) fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    let x = &x[..len];
    let len4 = len - len % 4;
    let mp4 = max_pitch - max_pitch % 4;
    let mut i = 0;
    while i < mp4 {
        let (mut c0, mut c1, mut c2, mut c3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut y0, mut y1, mut y2, mut y3) = (y[i], y[i + 1], y[i + 2], y[i + 3]);
        for (xc, yc) in x[..len4].chunks_exact(4).zip(y[i + 4..].chunks_exact(4)) {
            c0 += xc[0] * y0;
            c1 += xc[0] * y1;
            c2 += xc[0] * y2;
            c3 += xc[0] * y3;
            y0 = yc[0];
            c0 += xc[1] * y1;
            c1 += xc[1] * y2;
            c2 += xc[1] * y3;
            c3 += xc[1] * y0;
            y1 = yc[1];
            c0 += xc[2] * y2;
            c1 += xc[2] * y3;
            c2 += xc[2] * y0;
            c3 += xc[2] * y1;
            y2 = yc[2];
            c0 += xc[3] * y3;
            c1 += xc[3] * y0;
            c2 += xc[3] * y1;
            c3 += xc[3] * y2;
            y3 = yc[3];
        }
        for j in len4..len {
            c0 += x[j] * y[i + j];
            c1 += x[j] * y[i + 1 + j];
            c2 += x[j] * y[i + 2 + j];
            c3 += x[j] * y[i + 3 + j];
        }
        xcorr[i] = c0;
        xcorr[i + 1] = c1;
        xcorr[i + 2] = c2;
        xcorr[i + 3] = c3;
        i += 4;
    }
    while i < max_pitch {
        xcorr[i] = inner_prod(x, &y[i..], len);
        i += 1;
    }
}

/// `rnn_lpc`: Levinson–Durbin recursion. `ac` has length `p+1`.
pub(crate) fn lpc(out: &mut [f32], ac: &[f32], p: usize) {
    let mut error = ac[0];
    for v in out[..p].iter_mut() {
        *v = 0.0;
    }
    if ac[0] != 0.0 {
        for i in 0..p {
            let mut rr = 0.0f32;
            for j in 0..i {
                rr += out[j] * ac[i - j];
            }
            rr += ac[i + 1];
            let r = -rr / error;
            out[i] = r;
            for j in 0..(i + 1) >> 1 {
                let tmp1 = out[j];
                let tmp2 = out[i - 1 - j];
                out[j] = tmp1 + r * tmp2;
                out[i - 1 - j] = tmp2 + r * tmp1;
            }
            error -= r * r * error;
            if error < 0.001 * ac[0] {
                break;
            }
        }
    }
}

/// `rnn_autocorr` for the float, no-window, no-overlap case used by pitch
/// downsampling. Writes `ac[0..=lag]`.
pub(crate) fn autocorr(x: &[f32], ac: &mut [f32], lag: usize, n: usize) {
    let fast_n = n - lag;
    pitch_xcorr(x, x, &mut ac[..lag + 1], fast_n, lag + 1);
    for k in 0..=lag {
        let mut d = 0.0f32;
        for i in (k + fast_n)..n {
            d += x[i] * x[i - k];
        }
        ac[k] += d;
    }
}

/// `celt_fir5`: order-5 FIR applied in place (the only way RNNoise uses it).
pub(crate) fn fir5(x: &mut [f32], num: &[f32; 5], mem: &mut [f32; 5]) {
    let (n0, n1, n2, n3, n4) = (num[0], num[1], num[2], num[3], num[4]);
    let (mut m0, mut m1, mut m2, mut m3, mut m4) = (mem[0], mem[1], mem[2], mem[3], mem[4]);
    for xi_ref in x.iter_mut() {
        let xi = *xi_ref;
        let sum = xi + n0 * m0 + n1 * m1 + n2 * m2 + n3 * m3 + n4 * m4;
        m4 = m3;
        m3 = m2;
        m2 = m1;
        m1 = m0;
        m0 = xi;
        *xi_ref = sum;
    }
    *mem = [m0, m1, m2, m3, m4];
}
