//! FFT plan: the radix factors, the bit-reversal permutation, the twiddle table and
//! the per-stage (contiguous) twiddle views, plus the transform entry point.
//!
//! Everything here mirrors `kiss_fft.c` (float configuration): the factors come from
//! `kf_factor`, the twiddles are generated in double precision exactly as the C
//! generator does, and the transform is the scaled DFT that `rnn_fft_c` produces.

use crate::fft::dispatch;
use crate::fft::spectrum::Cpx;

const MAXFACTORS: usize = 8;

/// One butterfly stage, in the order `rnn_fft_impl` walks them.
pub(crate) struct Stage {
    /// Radix (2, 3, 4 or 5).
    pub p: usize,
    /// Length of the inner loop (`m` in `rnn_fft_impl`).
    pub m: usize,
    /// Number of groups (`fstride[i]`).
    pub n: usize,
    /// Group stride (`mm` in `rnn_fft_impl`).
    pub mm: usize,
    /// `p * m` twiddles, laid out so stage entry `k*m + j` equals C `tw[j*fstride*k]`.
    /// The values are copied from the plan table, so the arithmetic is identical.
    pub tw_re: Vec<f32>,
    pub tw_im: Vec<f32>,
    /// `tw[fstride*m].i` — the `epi3_i` constant of `bfly3`.
    pub epi3_i: f32,
    /// `tw[fstride*m]` and `tw[2*fstride*m]` — the `ya`/`yb` constants of `bfly5`.
    pub ya: Cpx,
    pub yb: Cpx,
}

/// FFT configuration: twiddles, bit-reversal table and radix factors.
pub struct KissFft {
    nfft: usize,
    pub(crate) scale: f32,
    /// Pairs of (radix p, m) — same layout as C `factors[2*MAXFACTORS]`.
    pub(crate) factors: [i16; 2 * MAXFACTORS],
    pub(crate) bitrev: Vec<i32>,
    pub(crate) twiddles: Vec<Cpx>,
    pub(crate) stages: Vec<Stage>,
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
            stages: Vec::new(),
        };
        st.compute_bitrev_table(0, 0, 1, 0, &mut bitrev);

        let stages = st.build_stages();
        KissFft {
            bitrev,
            stages,
            ..st
        }
    }

    #[allow(dead_code)]
    pub fn nfft(&self) -> usize {
        self.nfft
    }

    /// Forward FFT (matches `rnn_fft_c`): scales the input by `1/nfft`, applies the
    /// bit-reversal permutation into `fout`, then runs the in-place butterflies.
    /// `fin` and `fout` must both have length `nfft`.
    pub fn forward(&self, fin: &[Cpx], fout: &mut [Cpx]) {
        self.forward_scaled(fin, fout, self.scale);
    }

    /// As [`KissFft::forward`] but with a caller-chosen input scale, so the output is
    /// `scale · DFT(fin)`. Used by the legacy model, which normalises by the window
    /// energy (`wnorm`) rather than `1/nfft`.
    pub fn forward_scaled(&self, fin: &[Cpx], fout: &mut [Cpx], scale: f32) {
        debug_assert_eq!(fin.len(), self.nfft);
        debug_assert_eq!(fout.len(), self.nfft);
        for i in 0..self.nfft {
            let x = fin[i];
            let dst = self.bitrev[i] as usize;
            fout[dst] = Cpx {
                r: scale * x.r,
                i: scale * x.i,
            };
        }
        dispatch::process(&self.stages, fout);
    }

    // --- internal ---------------------------------------------------------

    fn build_stages(&self) -> Vec<Stage> {
        let factors = &self.factors;
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
        let mut stages = Vec::with_capacity(l);
        for i in (0..l).rev() {
            let m2 = if i != 0 { factors[2 * i - 1] as i32 } else { 1 };
            let p = factors[2 * i] as usize;
            let mm = m2 as usize;
            let mi = m as usize;
            let fs = fstride[i];
            let mut tw_re = vec![0.0f32; p * mi];
            let mut tw_im = vec![0.0f32; p * mi];
            for k in 0..p {
                for j in 0..mi {
                    let t = self.twiddles[j * fs * k];
                    tw_re[k * mi + j] = t.r;
                    tw_im[k * mi + j] = t.i;
                }
            }
            // `epi3_i` (radix 3) and `ya`/`yb` (radix 5) are read from the plan table at
            // `j == m`, which is outside the per-stage range. Other radices do not need
            // them, and for those `fs * m` can fall outside the table.
            let (epi3_i, ya, yb) = if p == 3 {
                let a = self.twiddles[fs * mi];
                (a.i, a, Cpx::default())
            } else if p == 5 {
                let a = self.twiddles[fs * mi];
                (0.0, a, self.twiddles[2 * fs * mi])
            } else {
                (0.0, Cpx::default(), Cpx::default())
            };
            stages.push(Stage {
                p,
                m: mi,
                n: fs,
                mm,
                tw_re,
                tw_im,
                epi3_i,
                ya,
                yb,
            });
            m = m2;
        }
        stages
    }

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
