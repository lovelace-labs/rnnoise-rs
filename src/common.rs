//! Shared DSP primitives: analysis window, DCT, band energy/correlation,
//! band-gain interpolation, the high-pass biquad and the FFT-based
//! forward/inverse transforms. All ported from `denoise.c` / `rnnoise_tables.c`.

use crate::fft::{Cpx, KissFft};
use crate::{EBAND20MS, FRAME_SIZE, FREQ_SIZE, NB_BANDS, WINDOW_SIZE};

/// Precomputed tables + the FFT plan, built once and shared by every
/// [`crate::DenoiseState`].
pub(crate) struct Common {
    /// Half analysis window, length `FRAME_SIZE`, applied symmetrically.
    half_window: [f32; FRAME_SIZE],
    /// `NB_BANDS x NB_BANDS` DCT matrix (row-major, as in `rnn_dct_table`).
    dct_table: [f32; NB_BANDS * NB_BANDS],
    fft: KissFft,
}

/// `sqrt(2/22)` — the DCT normalisation constant from `denoise.c` (the `22` is
/// a historical hold-over from the old 22-band model; kept for weight compat).
const DCT_SCALE: f64 = 0.301_511_344_577_763_5; // (2.0/22.0).sqrt()

impl Common {
    pub(crate) fn new() -> Self {
        use std::f64::consts::PI;

        let mut half_window = [0.0f32; FRAME_SIZE];
        for i in 0..FRAME_SIZE {
            // sin(.5*pi*sin(.5*pi*(i+.5)/N)^2), computed in f64 like the C generator.
            let inner = (0.5 * PI * (i as f64 + 0.5) / FRAME_SIZE as f64).sin();
            half_window[i] = (0.5 * PI * inner * inner).sin() as f32;
        }

        let mut dct_table = [0.0f32; NB_BANDS * NB_BANDS];
        for i in 0..NB_BANDS {
            for j in 0..NB_BANDS {
                let mut v = ((i as f64 + 0.5) * j as f64 * PI / NB_BANDS as f64).cos();
                if j == 0 {
                    v *= 0.5f64.sqrt();
                }
                dct_table[i * NB_BANDS + j] = v as f32;
            }
        }

        Common {
            half_window,
            dct_table,
            fft: KissFft::new(WINDOW_SIZE),
        }
    }

    /// Type-II-ish DCT used for cepstral features (`dct` in `denoise.c`).
    pub(crate) fn dct(&self, out: &mut [f32], input: &[f32]) {
        for i in 0..NB_BANDS {
            let mut sum = 0.0f32;
            for j in 0..NB_BANDS {
                sum += input[j] * self.dct_table[j * NB_BANDS + i];
            }
            out[i] = (sum as f64 * DCT_SCALE) as f32;
        }
    }

    /// Apply the symmetric analysis window to a `WINDOW_SIZE` buffer in place.
    pub(crate) fn apply_window(&self, x: &mut [f32]) {
        for i in 0..FRAME_SIZE {
            x[i] *= self.half_window[i];
            x[WINDOW_SIZE - 1 - i] *= self.half_window[i];
        }
    }

    /// Forward transform: window-domain real signal → spectrum (`forward_transform`).
    /// `out` holds the first `FREQ_SIZE` bins. Includes the kiss `1/nfft` scaling.
    pub(crate) fn forward_transform(&self, out: &mut [Cpx], input: &[f32]) {
        let mut x = [Cpx::default(); WINDOW_SIZE];
        for i in 0..WINDOW_SIZE {
            x[i] = Cpx::new(input[i], 0.0);
        }
        let mut y = [Cpx::default(); WINDOW_SIZE];
        self.fft.forward(&x, &mut y);
        out[..FREQ_SIZE].copy_from_slice(&y[..FREQ_SIZE]);
    }

    /// Inverse transform via the forward FFT on a conjugate-symmetric spectrum,
    /// then scaled by `WINDOW_SIZE` (`inverse_transform`).
    pub(crate) fn inverse_transform(&self, out: &mut [f32], input: &[Cpx]) {
        let mut x = [Cpx::default(); WINDOW_SIZE];
        x[..FREQ_SIZE].copy_from_slice(&input[..FREQ_SIZE]);
        for i in FREQ_SIZE..WINDOW_SIZE {
            x[i] = Cpx::new(x[WINDOW_SIZE - i].r, -x[WINDOW_SIZE - i].i);
        }
        let mut y = [Cpx::default(); WINDOW_SIZE];
        self.fft.forward(&x, &mut y);
        // Output in reverse order for the IFFT.
        out[0] = WINDOW_SIZE as f32 * y[0].r;
        for i in 1..WINDOW_SIZE {
            out[i] = WINDOW_SIZE as f32 * y[WINDOW_SIZE - i].r;
        }
    }
}

/// `compute_band_energy`: aggregate `|X|^2` into `NB_BANDS` triangular bands.
pub(crate) fn compute_band_energy(band_e: &mut [f32], x: &[Cpx]) {
    let mut sum = [0.0f32; NB_BANDS + 2];
    for i in 0..NB_BANDS + 1 {
        let band_size = EBAND20MS[i + 1] - EBAND20MS[i];
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let xx = x[EBAND20MS[i] + j];
            let tmp = xx.r * xx.r + xx.i * xx.i;
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[1] = (sum[0] + sum[1]) * 2.0 / 3.0;
    sum[NB_BANDS] = (sum[NB_BANDS] + sum[NB_BANDS + 1]) * 2.0 / 3.0;
    band_e[..NB_BANDS].copy_from_slice(&sum[1..=NB_BANDS]);
}

/// `compute_band_corr`: aggregate `Re(X conj-correlated with P)` into bands.
pub(crate) fn compute_band_corr(band_e: &mut [f32], x: &[Cpx], p: &[Cpx]) {
    let mut sum = [0.0f32; NB_BANDS + 2];
    for i in 0..NB_BANDS + 1 {
        let band_size = EBAND20MS[i + 1] - EBAND20MS[i];
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = EBAND20MS[i] + j;
            let tmp = x[idx].r * p[idx].r + x[idx].i * p[idx].i;
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[1] = (sum[0] + sum[1]) * 2.0 / 3.0;
    sum[NB_BANDS] = (sum[NB_BANDS] + sum[NB_BANDS + 1]) * 2.0 / 3.0;
    band_e[..NB_BANDS].copy_from_slice(&sum[1..=NB_BANDS]);
}

/// `interp_band_gain`: expand per-band gains to per-bin gains by linear
/// interpolation. Writes bins `0..EBAND20MS[NB_BANDS+1]` (=400) and zeroes the
/// rest of `g` — matching the upstream behaviour where callers pre-zero the
/// buffer and bins above 20 kHz stay silent.
pub(crate) fn interp_band_gain(g: &mut [f32], band_e: &[f32]) {
    for v in g[..FREQ_SIZE].iter_mut() {
        *v = 0.0;
    }
    for i in 1..NB_BANDS {
        let band_size = EBAND20MS[i + 1] - EBAND20MS[i];
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            g[EBAND20MS[i] + j] = (1.0 - frac) * band_e[i - 1] + frac * band_e[i];
        }
    }
    for j in 0..EBAND20MS[1] {
        g[j] = band_e[0];
    }
    for j in EBAND20MS[NB_BANDS]..EBAND20MS[NB_BANDS + 1] {
        g[j] = band_e[NB_BANDS - 1];
    }
}

/// `rnn_biquad`: direct-form-II transposed biquad, with the double-precision
/// state update used by upstream's DC-removal high-pass.
pub(crate) fn biquad(y: &mut [f32], mem: &mut [f32; 2], x: &[f32], b: &[f32; 2], a: &[f32; 2]) {
    let (b0, b1) = (b[0] as f64, b[1] as f64);
    let (a0, a1) = (a[0] as f64, a[1] as f64);
    for i in 0..x.len() {
        let xi = x[i] as f64;
        let yi = x[i] + mem[0];
        let yid = yi as f64;
        mem[0] = (mem[1] as f64 + (b0 * xi - a0 * yid)) as f32;
        mem[1] = (b1 * xi - a1 * yid) as f32;
        y[i] = yi;
    }
}
