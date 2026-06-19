//! Pitch analysis ported from `pitch.c` (float configuration): downsampling +
//! LPC whitening, coarse/fine cross-correlation search, and pitch-doubling
//! removal.

use crate::celt_lpc::{autocorr, dual_inner_prod, fir5, inner_prod, lpc, pitch_xcorr};
use crate::{PITCH_FRAME_SIZE, PITCH_MAX_PERIOD};

/// `rnn_pitch_downsample` for a single channel: 2× decimate `x` (length `len`)
/// into `x_lp` (length `len/2`), then apply a short LPC whitening filter.
pub(crate) fn pitch_downsample(x: &[f32], x_lp: &mut [f32], len: usize) {
    let half = len >> 1;
    for i in 1..half {
        x_lp[i] = 0.5 * (0.5 * (x[2 * i - 1] + x[2 * i + 1]) + x[2 * i]);
    }
    x_lp[0] = 0.5 * (0.5 * x[1] + x[0]);

    let mut ac = [0.0f32; 5];
    autocorr(&x_lp[..half], &mut ac, 4, half);

    // Noise floor at -40 dB.
    ac[0] *= 1.0001;
    // Lag windowing.
    for i in 1..=4 {
        ac[i] -= ac[i] * (0.008 * i as f32) * (0.008 * i as f32);
    }

    let mut lpc_coef = [0.0f32; 4];
    lpc(&mut lpc_coef, &ac, 4);

    let mut tmp = 1.0f32;
    for c in lpc_coef.iter_mut() {
        tmp *= 0.9;
        *c *= tmp;
    }

    // Add a zero to the whitening filter.
    let c1 = 0.8f32;
    let lpc2 = [
        lpc_coef[0] + 0.8,
        lpc_coef[1] + c1 * lpc_coef[0],
        lpc_coef[2] + c1 * lpc_coef[1],
        lpc_coef[3] + c1 * lpc_coef[2],
        c1 * lpc_coef[3],
    ];
    let mut mem = [0.0f32; 5];
    fir5(&mut x_lp[..half], &lpc2, &mut mem);
}

/// `find_best_pitch`: pick the two best lags by normalised correlation.
fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize, best_pitch: &mut [usize; 2]) {
    let mut syy = 1.0f32;
    let mut best_num = [-1.0f32; 2];
    let mut best_den = [0.0f32; 2];
    best_pitch[0] = 0;
    best_pitch[1] = 1;
    for j in 0..len {
        syy += y[j] * y[j];
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let xcorr16 = xcorr[i] * 1e-12;
            let num = xcorr16 * xcorr16;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        syy = syy.max(1.0);
    }
}

/// `rnn_pitch_search`: coarse (4× decimated) then fine (2× decimated) search,
/// returning the estimated lag (`*pitch`).
pub(crate) fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> i32 {
    let lag = len + max_pitch;
    let mut x_lp4 = [0.0f32; PITCH_FRAME_SIZE >> 2];
    let mut y_lp4 = [0.0f32; (PITCH_FRAME_SIZE + PITCH_MAX_PERIOD) >> 2];
    let mut xcorr = [0.0f32; PITCH_MAX_PERIOD >> 1];
    let mut best_pitch = [0usize; 2];

    // Downsample by 2 again.
    for j in 0..len >> 2 {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..lag >> 2 {
        y_lp4[j] = y[2 * j];
    }

    // Coarse search with 4× decimation.
    pitch_xcorr(&x_lp4[..len >> 2], &y_lp4, &mut xcorr[..max_pitch >> 2], len >> 2, max_pitch >> 2);
    find_best_pitch(&xcorr, &y_lp4, len >> 2, max_pitch >> 2, &mut best_pitch);

    // Finer search with 2× decimation.
    for i in 0..max_pitch >> 1 {
        xcorr[i] = 0.0;
        if (i as i32 - 2 * best_pitch[0] as i32).abs() > 2
            && (i as i32 - 2 * best_pitch[1] as i32).abs() > 2
        {
            continue;
        }
        let sum = inner_prod(x_lp, &y[i..], len >> 1);
        xcorr[i] = sum.max(-1.0);
    }
    find_best_pitch(&xcorr, y, len >> 1, max_pitch >> 1, &mut best_pitch);

    // Refine by pseudo-interpolation.
    let offset = if best_pitch[0] > 0 && best_pitch[0] < (max_pitch >> 1) - 1 {
        let a = xcorr[best_pitch[0] - 1];
        let b = xcorr[best_pitch[0]];
        let c = xcorr[best_pitch[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            1
        } else if (a - c) > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    2 * best_pitch[0] as i32 - offset
}

#[inline]
fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    (xy as f64 / ((1.0f32 + xx * yy) as f64).sqrt()) as f32
}

const SECOND_CHECK: [i32; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// `rnn_remove_doubling`: detect and correct pitch period doubling/halving.
/// `x` is the half-rate buffer; `t0` is the in/out pitch estimate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn remove_doubling(
    x: &[f32],
    maxperiod: i32,
    minperiod: i32,
    n: i32,
    t0: &mut i32,
    prev_period: i32,
    prev_gain: f32,
) -> f32 {
    let minperiod0 = minperiod;
    let maxperiod = maxperiod / 2;
    let minperiod = minperiod / 2;
    *t0 /= 2;
    let prev_period = prev_period / 2;
    let n = n / 2;
    // C does `x += maxperiod`; emulate with an explicit base offset.
    let xb = maxperiod as usize;
    // Helpers translating C's `x[k]` (and negative `x[-k]`) into our slice.
    let at = |k: i32| x[(xb as i32 + k) as usize];

    if *t0 >= maxperiod {
        *t0 = maxperiod - 1;
    }
    let mut t = *t0;
    let t0v = *t0;

    let (xx, xy) = dual_inner_prod(
        &x[xb..],
        &x[xb..],
        &x[(xb as i32 - t0v) as usize..],
        n as usize,
    );

    // maxperiod here is PITCH_MAX_PERIOD/2 = 384, so 385 entries suffice.
    let mut yy_lookup = [0.0f32; (PITCH_MAX_PERIOD / 2) + 1];
    let yy_lookup = &mut yy_lookup[..maxperiod as usize + 1];
    yy_lookup[0] = xx;
    let mut yy = xx;
    for i in 1..=maxperiod {
        yy = yy + at(-i) * at(-i) - at(n - i) * at(n - i);
        yy_lookup[i as usize] = yy.max(0.0);
    }
    let mut yy = yy_lookup[t0v as usize];
    let mut best_xy = xy;
    let mut best_yy = yy;
    let g0 = compute_pitch_gain(xy, xx, yy);
    let mut g = g0;

    for k in 2..=15i32 {
        let t1 = (2 * t0v + k) / (2 * k);
        if t1 < minperiod {
            break;
        }
        let t1b = if k == 2 {
            if t1 + t0v > maxperiod {
                t0v
            } else {
                t0v + t1
            }
        } else {
            (2 * SECOND_CHECK[k as usize] * t0v + k) / (2 * k)
        };
        let (mut xy_k, xy2) = dual_inner_prod(
            &x[xb..],
            &x[(xb as i32 - t1) as usize..],
            &x[(xb as i32 - t1b) as usize..],
            n as usize,
        );
        xy_k = 0.5 * (xy_k + xy2);
        yy = 0.5 * (yy_lookup[t1 as usize] + yy_lookup[t1b as usize]);
        let g1 = compute_pitch_gain(xy_k, xx, yy);

        let cont = if (t1 - prev_period).abs() <= 1 {
            prev_gain
        } else if (t1 - prev_period).abs() <= 2 && 5 * k * k < t0v {
            0.5 * prev_gain
        } else {
            0.0
        };
        let mut thresh = (0.7 * g0 - cont).max(0.3);
        if t1 < 3 * minperiod {
            thresh = (0.85 * g0 - cont).max(0.4);
        } else if t1 < 2 * minperiod {
            thresh = (0.9 * g0 - cont).max(0.5);
        }
        if g1 > thresh {
            best_xy = xy_k;
            best_yy = yy;
            t = t1;
            g = g1;
        }
    }
    best_xy = best_xy.max(0.0);
    let pg = if best_yy <= best_xy {
        1.0
    } else {
        best_xy / (best_yy + 1.0)
    };

    let mut xcorr = [0.0f32; 3];
    for k in 0..3i32 {
        xcorr[k as usize] = inner_prod(&x[xb..], &x[(xb as i32 - (t + k - 1)) as usize..], n as usize);
    }
    let offset = if (xcorr[2] - xcorr[0]) > 0.7 * (xcorr[1] - xcorr[0]) {
        1
    } else if (xcorr[0] - xcorr[2]) > 0.7 * (xcorr[1] - xcorr[2]) {
        -1
    } else {
        0
    };
    let pg = pg.min(g);
    *t0 = 2 * t + offset;
    if *t0 < minperiod0 {
        *t0 = minperiod0;
    }
    pg
}
