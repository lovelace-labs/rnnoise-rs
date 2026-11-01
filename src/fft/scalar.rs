//! The portable reference butterflies — operation-for-operation the same expression
//! sequence as `rnn_fft_impl` in `kiss_fft.c`. This is the definition the vector tiers
//! in [`crate::fft::x86_64`] must match bit for bit.

use crate::fft::plan::Stage;
use crate::fft::spectrum::Cpx;

pub(crate) fn process(stages: &[Stage], f: &mut [Cpx]) {
    for s in stages {
        stage(s, f);
    }
}

pub(crate) fn stage(s: &Stage, f: &mut [Cpx]) {
    match s.p {
        2 => bfly2(s, f),
        4 => bfly4(s, f),
        3 => bfly3(s, f),
        5 => bfly5(s, f),
        p => unreachable!("unsupported radix {p}"),
    }
}

#[inline(always)]
fn c_mul(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}

#[inline(always)]
fn c_add(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}

#[inline(always)]
fn c_sub(a: Cpx, b: Cpx) -> Cpx {
    Cpx {
        r: a.r - b.r,
        i: a.i - b.i,
    }
}

fn bfly2(s: &Stage, f: &mut [Cpx]) {
    let m = s.m;
    let n = s.n;
    if m == 1 {
        for i in 0..n {
            let base = 2 * i;
            let t = f[base + 1];
            f[base + 1] = c_sub(f[base], t);
            f[base] = c_add(f[base], t);
        }
    } else {
        // m == 4: radix-2 right after a radix-4.
        let tw = 0.707_106_77_f32;
        for i in 0..n {
            let base = 8 * i;
            let mut t = f[base + 4];
            f[base + 4] = c_sub(f[base], t);
            f[base] = c_add(f[base], t);

            t.r = (f[base + 5].r + f[base + 5].i) * tw;
            t.i = (f[base + 5].i - f[base + 5].r) * tw;
            f[base + 5] = c_sub(f[base + 1], t);
            f[base + 1] = c_add(f[base + 1], t);

            t.r = f[base + 6].i;
            t.i = -f[base + 6].r;
            f[base + 6] = c_sub(f[base + 2], t);
            f[base + 2] = c_add(f[base + 2], t);

            t.r = (f[base + 7].i - f[base + 7].r) * tw;
            t.i = -((f[base + 7].i + f[base + 7].r) * tw);
            f[base + 7] = c_sub(f[base + 3], t);
            f[base + 3] = c_add(f[base + 3], t);
        }
    }
}

pub(crate) fn bfly4(s: &Stage, f: &mut [Cpx]) {
    let m = s.m;
    let mm = s.mm;
    if m == 1 {
        for i in 0..s.n {
            let base = 4 * i;
            let scratch0 = c_sub(f[base], f[base + 2]);
            f[base] = c_add(f[base], f[base + 2]);
            let scratch1 = c_add(f[base + 1], f[base + 3]);
            f[base + 2] = c_sub(f[base], scratch1);
            f[base] = c_add(f[base], scratch1);
            let scratch1 = c_sub(f[base + 1], f[base + 3]);
            f[base + 1] = Cpx {
                r: scratch0.r + scratch1.i,
                i: scratch0.i - scratch1.r,
            };
            f[base + 3] = Cpx {
                r: scratch0.r - scratch1.i,
                i: scratch0.i + scratch1.r,
            };
        }
        return;
    }
    let m2 = 2 * m;
    let m3 = 3 * m;
    for i in 0..s.n {
        for j in 0..m {
            let idx = i * mm + j;
            let tw1 = Cpx::new(s.tw_re[m + j], s.tw_im[m + j]);
            let tw2 = Cpx::new(s.tw_re[m2 + j], s.tw_im[m2 + j]);
            let tw3 = Cpx::new(s.tw_re[m3 + j], s.tw_im[m3 + j]);
            let s0 = c_mul(f[idx + m], tw1);
            let s1 = c_mul(f[idx + m2], tw2);
            let s2 = c_mul(f[idx + m3], tw3);
            let s5 = c_sub(f[idx], s1);
            f[idx] = c_add(f[idx], s1);
            let s3 = c_add(s0, s2);
            let s4 = c_sub(s0, s2);
            f[idx + m2] = c_sub(f[idx], s3);
            f[idx] = c_add(f[idx], s3);
            f[idx + m] = Cpx {
                r: s5.r + s4.i,
                i: s5.i - s4.r,
            };
            f[idx + m3] = Cpx {
                r: s5.r - s4.i,
                i: s5.i + s4.r,
            };
        }
    }
}

pub(crate) fn bfly3(s: &Stage, f: &mut [Cpx]) {
    let m = s.m;
    let mm = s.mm;
    let m2 = 2 * m;
    let epi3_i = s.epi3_i;
    for i in 0..s.n {
        for j in 0..m {
            let idx = i * mm + j;
            let tw1 = Cpx::new(s.tw_re[m + j], s.tw_im[m + j]);
            let tw2 = Cpx::new(s.tw_re[m2 + j], s.tw_im[m2 + j]);
            let s1 = c_mul(f[idx + m], tw1);
            let s2 = c_mul(f[idx + m2], tw2);
            let s3 = c_add(s1, s2);
            let mut s0 = c_sub(s1, s2);

            f[idx + m].r = f[idx].r - 0.5 * s3.r;
            f[idx + m].i = f[idx].i - 0.5 * s3.i;

            s0.r *= epi3_i;
            s0.i *= epi3_i;

            f[idx] = c_add(f[idx], s3);

            f[idx + m2].r = f[idx + m].r + s0.i;
            f[idx + m2].i = f[idx + m].i - s0.r;

            f[idx + m].r -= s0.i;
            f[idx + m].i += s0.r;
        }
    }
}

pub(crate) fn bfly5(s: &Stage, f: &mut [Cpx]) {
    let m = s.m;
    let mm = s.mm;
    let m2 = 2 * m;
    let m3 = 3 * m;
    let m4 = 4 * m;
    let ya = s.ya;
    let yb = s.yb;
    for i in 0..s.n {
        let base = i * mm;
        for j in 0..m {
            let s0 = f[base + j];
            let tw1 = Cpx::new(s.tw_re[m + j], s.tw_im[m + j]);
            let tw2 = Cpx::new(s.tw_re[m2 + j], s.tw_im[m2 + j]);
            let tw3 = Cpx::new(s.tw_re[m3 + j], s.tw_im[m3 + j]);
            let tw4 = Cpx::new(s.tw_re[m4 + j], s.tw_im[m4 + j]);
            let s1 = c_mul(f[base + m + j], tw1);
            let s2 = c_mul(f[base + m2 + j], tw2);
            let s3 = c_mul(f[base + m3 + j], tw3);
            let s4 = c_mul(f[base + m4 + j], tw4);

            let s7 = c_add(s1, s4);
            let s10 = c_sub(s1, s4);
            let s8 = c_add(s2, s3);
            let s9 = c_sub(s2, s3);

            f[base + j].r += s7.r + s8.r;
            f[base + j].i += s7.i + s8.i;

            let s5 = Cpx {
                r: s0.r + (s7.r * ya.r + s8.r * yb.r),
                i: s0.i + (s7.i * ya.r + s8.i * yb.r),
            };
            let s6 = Cpx {
                r: s10.i * ya.i + s9.i * yb.i,
                i: -(s10.r * ya.i + s9.r * yb.i),
            };
            f[base + m + j] = c_sub(s5, s6);
            f[base + m4 + j] = c_add(s5, s6);

            let s11 = Cpx {
                r: s0.r + (s7.r * yb.r + s8.r * ya.r),
                i: s0.i + (s7.i * yb.r + s8.i * ya.r),
            };
            let s12 = Cpx {
                r: s9.i * ya.i - s10.i * yb.i,
                i: s10.r * yb.i - s9.r * ya.i,
            };
            f[base + m2 + j] = c_add(s11, s12);
            f[base + m3 + j] = c_sub(s11, s12);
        }
    }
}
