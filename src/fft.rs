//! Faithful port of the Opus/RNNoise variant of KISS-FFT (`kiss_fft.c`,
//! float configuration). The butterfly operations, twiddle generation and
//! bit-reversal table are replicated exactly so the transform is numerically
//! equivalent to the C reference (`rnn_fft_c`), including the `1/nfft` scaling
//! that the C code applies on the forward transform.
//!
//! RNNoise performs its inverse transform by calling the *forward* FFT on a
//! conjugate-symmetric spectrum (see [`crate::denoise`]), so only the forward
//! transform is needed here.

/// Complex value, matching C `kiss_fft_cpx` ({ float r; float i; }).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cpx {
    pub r: f32,
    pub i: f32,
}

impl Cpx {
    #[inline]
    pub const fn new(r: f32, i: f32) -> Self {
        Cpx { r, i }
    }
}

const MAXFACTORS: usize = 8;

/// FFT configuration: twiddles, bit-reversal table and radix factors.
pub struct KissFft {
    nfft: usize,
    scale: f32,
    /// Pairs of (radix p, m) — same layout as C `factors[2*MAXFACTORS]`.
    factors: [i16; 2 * MAXFACTORS],
    bitrev: Vec<i32>,
    twiddles: Vec<Cpx>,
}

#[inline(always)]
fn c_mul(a: Cpx, b: Cpx) -> Cpx {
    // C_MUL: m.r = a.r*b.r - a.i*b.i; m.i = a.r*b.i + a.i*b.r
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

impl KissFft {
    /// Build an FFT state for `nfft` points. `nfft` must factor into 2/3/4/5.
    pub fn new(nfft: usize) -> Self {
        let mut factors = [0i16; 2 * MAXFACTORS];
        assert!(
            kf_factor(nfft as i32, &mut factors),
            "unsupported FFT size {nfft}"
        );

        let mut twiddles = vec![Cpx::default(); nfft];
        compute_twiddles(&mut twiddles, nfft);

        let mut bitrev = vec![0i32; nfft];
        let st = KissFft {
            nfft,
            scale: 1.0 / nfft as f32,
            factors,
            bitrev: Vec::new(),
            twiddles,
        };
        st.compute_bitrev_table(0, 0, 1, 0, &mut bitrev);

        KissFft { bitrev, ..st }
    }

    #[allow(dead_code)]
    pub fn nfft(&self) -> usize {
        self.nfft
    }

    /// Forward FFT (matches `rnn_fft_c`): scales the input by `1/nfft`,
    /// applies the bit-reversal permutation into `fout`, then runs the
    /// in-place butterflies. `fin` and `fout` must both have length `nfft`.
    pub fn forward(&self, fin: &[Cpx], fout: &mut [Cpx]) {
        debug_assert_eq!(fin.len(), self.nfft);
        debug_assert_eq!(fout.len(), self.nfft);
        let scale = self.scale;
        for i in 0..self.nfft {
            let x = fin[i];
            let dst = self.bitrev[i] as usize;
            fout[dst] = Cpx {
                r: scale * x.r,
                i: scale * x.i,
            };
        }
        self.process(fout);
    }

    // --- internal ---------------------------------------------------------

    fn compute_bitrev_table(
        &self,
        fout: i32,
        f: usize,
        fstride: usize,
        fi: usize,
        bitrev: &mut [i32],
    ) {
        let p = self.factors[2 * fi] as i32;
        let m = self.factors[2 * fi + 1] as i32;
        if m == 1 {
            for j in 0..p {
                bitrev[f + (j as usize) * fstride] = fout + j;
            }
        } else {
            for j in 0..p {
                self.compute_bitrev_table(
                    fout + j * m,
                    f + (j as usize) * fstride,
                    fstride * (p as usize),
                    fi + 1,
                    bitrev,
                );
            }
        }
    }

    /// Port of `rnn_fft_impl`.
    fn process(&self, fout: &mut [Cpx]) {
        let factors = &self.factors;
        let shift = 0usize; // st->shift is -1 here, clamped to 0.

        let mut fstride = [0usize; MAXFACTORS + 1];
        fstride[0] = 1;
        let mut l = 0usize;
        let mut m;
        loop {
            let p = factors[2 * l] as usize;
            m = factors[2 * l + 1] as i32;
            fstride[l + 1] = fstride[l] * p;
            l += 1;
            if m == 1 {
                break;
            }
        }
        m = factors[2 * l - 1] as i32;
        for i in (0..l).rev() {
            let m2 = if i != 0 { factors[2 * i - 1] as i32 } else { 1 };
            match factors[2 * i] {
                2 => bfly2(fout, m, fstride[i] as i32),
                4 => bfly4(
                    fout,
                    fstride[i] << shift,
                    &self.twiddles,
                    m,
                    fstride[i] as i32,
                    m2,
                ),
                3 => bfly3(
                    fout,
                    fstride[i] << shift,
                    &self.twiddles,
                    m,
                    fstride[i] as i32,
                    m2,
                ),
                5 => bfly5(
                    fout,
                    fstride[i] << shift,
                    &self.twiddles,
                    m,
                    fstride[i] as i32,
                    m2,
                ),
                p => unreachable!("unsupported radix {p}"),
            }
            m = m2;
        }
    }
}

fn compute_twiddles(twiddles: &mut [Cpx], nfft: usize) {
    const PI: f64 = std::f64::consts::PI;
    for (i, t) in twiddles.iter_mut().enumerate() {
        let phase = (-2.0 * PI / nfft as f64) * i as f64;
        t.r = phase.cos() as f32;
        t.i = phase.sin() as f32;
    }
}

/// Port of `kf_factor`: factor `n` into 4s, then 2/3/5, reversing so radix-4
/// stages land at the end (kiss's "fast degenerate case").
fn kf_factor(n: i32, facbuf: &mut [i16]) -> bool {
    let mut p = 4i32;
    let mut stages = 0usize;
    let nbak = n;
    let mut n = n;
    loop {
        while n % p != 0 {
            match p {
                4 => p = 2,
                2 => p = 3,
                _ => p += 2,
            }
            if p > 32000 || p * p > n {
                p = n;
            }
        }
        n /= p;
        if p > 5 {
            return false;
        }
        facbuf[2 * stages] = p as i16;
        if p == 2 && stages > 1 {
            facbuf[2 * stages] = 4;
            facbuf[2] = 2;
        }
        stages += 1;
        if n <= 1 {
            break;
        }
    }
    let mut n = nbak;
    // Reverse the order to get radix-4 at the end.
    for i in 0..stages / 2 {
        facbuf.swap(2 * i, 2 * (stages - i - 1));
    }
    for i in 0..stages {
        n /= facbuf[2 * i] as i32;
        facbuf[2 * i + 1] = n as i16;
    }
    true
}

fn bfly2(f: &mut [Cpx], m: i32, n: i32) {
    if m == 1 {
        for i in 0..n as usize {
            let base = 2 * i;
            let t = f[base + 1];
            f[base + 1] = c_sub(f[base], t);
            f[base] = c_add(f[base], t);
        }
    } else {
        // m == 4: radix-2 right after a radix-4.
        let tw = 0.707_106_77_f32;
        for i in 0..n as usize {
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

fn bfly4(f: &mut [Cpx], fstride: usize, tw: &[Cpx], m: i32, n: i32, mm: i32) {
    let m = m as usize;
    let mm = mm as usize;
    if m == 1 {
        for i in 0..n as usize {
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
    } else {
        let m2 = 2 * m;
        let m3 = 3 * m;
        for i in 0..n as usize {
            for j in 0..m {
                let idx = i * mm + j;
                let tw1 = tw[j * fstride];
                let tw2 = tw[j * 2 * fstride];
                let tw3 = tw[j * 3 * fstride];
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
}

fn bfly3(f: &mut [Cpx], fstride: usize, tw: &[Cpx], m: i32, n: i32, mm: i32) {
    let m = m as usize;
    let mm = mm as usize;
    let m2 = 2 * m;
    let epi3_i = tw[fstride * m].i;
    for i in 0..n as usize {
        for u in 0..m {
            let idx = i * mm + u;
            let tw1 = tw[u * fstride];
            let tw2 = tw[u * 2 * fstride];
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

#[allow(clippy::too_many_arguments)]
fn bfly5(f: &mut [Cpx], fstride: usize, tw: &[Cpx], m: i32, n: i32, mm: i32) {
    let m = m as usize;
    let mm = mm as usize;
    let ya = tw[fstride * m];
    let yb = tw[fstride * 2 * m];
    for i in 0..n as usize {
        let base = i * mm;
        let (f0, f1, f2, f3, f4) = (base, base + m, base + 2 * m, base + 3 * m, base + 4 * m);
        for u in 0..m {
            let s0 = f[f0 + u];
            let s1 = c_mul(f[f1 + u], tw[u * fstride]);
            let s2 = c_mul(f[f2 + u], tw[2 * u * fstride]);
            let s3 = c_mul(f[f3 + u], tw[3 * u * fstride]);
            let s4 = c_mul(f[f4 + u], tw[4 * u * fstride]);

            let s7 = c_add(s1, s4);
            let s10 = c_sub(s1, s4);
            let s8 = c_add(s2, s3);
            let s9 = c_sub(s2, s3);

            f[f0 + u].r += s7.r + s8.r;
            f[f0 + u].i += s7.i + s8.i;

            let s5 = Cpx {
                r: s0.r + (s7.r * ya.r + s8.r * yb.r),
                i: s0.i + (s7.i * ya.r + s8.i * yb.r),
            };
            let s6 = Cpx {
                r: s10.i * ya.i + s9.i * yb.i,
                i: -(s10.r * ya.i + s9.r * yb.i),
            };
            f[f1 + u] = c_sub(s5, s6);
            f[f4 + u] = c_add(s5, s6);

            let s11 = Cpx {
                r: s0.r + (s7.r * yb.r + s8.r * ya.r),
                i: s0.i + (s7.i * yb.r + s8.i * ya.r),
            };
            let s12 = Cpx {
                r: s9.i * ya.i - s10.i * yb.i,
                i: s10.r * yb.i - s9.r * ya.i,
            };
            f[f2 + u] = c_add(s11, s12);
            f[f3 + u] = c_sub(s11, s12);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(input: &[Cpx], scale: f32) -> Vec<Cpx> {
        let n = input.len();
        let mut out = vec![Cpx::default(); n];
        for (k, o) in out.iter_mut().enumerate() {
            let mut acc = Cpx::default();
            for (j, x) in input.iter().enumerate() {
                let phase = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                let (s, c) = phase.sin_cos();
                let (c, s) = (c as f32, s as f32);
                acc.r += x.r * c - x.i * s;
                acc.i += x.r * s + x.i * c;
            }
            *o = Cpx {
                r: acc.r * scale,
                i: acc.i * scale,
            };
        }
        out
    }

    #[test]
    fn fft_matches_naive_dft_960() {
        let n = 960;
        let fft = KissFft::new(n);
        // deterministic pseudo-random real input
        let mut state = 12345u32;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / 16_777_216.0 - 0.5
        };
        let input: Vec<Cpx> = (0..n).map(|_| Cpx::new(next(), 0.0)).collect();
        let mut out = vec![Cpx::default(); n];
        fft.forward(&input, &mut out);
        let reference = naive_dft(&input, 1.0 / n as f32);
        let mut max_err = 0.0f32;
        for (a, b) in out.iter().zip(&reference) {
            max_err = max_err.max((a.r - b.r).abs()).max((a.i - b.i).abs());
        }
        assert!(max_err < 1e-5, "max FFT error {max_err}");
    }

    #[test]
    fn factors_960() {
        let mut fac = [0i16; 16];
        assert!(kf_factor(960, &mut fac));
        // 960 = 5 * 3 * 4 * 4 * 4 (processing order)
        assert_eq!(&fac[0..10], &[5, 192, 3, 64, 4, 16, 4, 4, 4, 1]);
    }
}
