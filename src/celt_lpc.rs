//! LPC analysis + inner-product kernels, ported from `celt_lpc.c` / `pitch.h`
//! (float configuration, so the fixed-point shift macros are all identities).

/// `celt_inner_prod`: `sum_{i<n} x[i] * y[i]`, accumulated in order.
#[inline]
pub(crate) fn inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    let mut xy = 0.0f32;
    for i in 0..n {
        xy += x[i] * y[i];
    }
    xy
}

/// `dual_inner_prod`: two correlations sharing the `x` operand.
#[inline]
pub(crate) fn dual_inner_prod(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    let mut a = 0.0f32;
    let mut b = 0.0f32;
    for i in 0..n {
        a += x[i] * y01[i];
        b += x[i] * y02[i];
    }
    (a, b)
}

/// `rnn_pitch_xcorr`: `xcorr[i] = sum_{j<len} x[j] * y[i+j]` for `i in 0..max_pitch`.
/// The C version unrolls 4 lags at a time but accumulates in `j` order, so this
/// straightforward loop is numerically identical.
pub(crate) fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    for i in 0..max_pitch {
        xcorr[i] = inner_prod(x, &y[i..], len);
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
