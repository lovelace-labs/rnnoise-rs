//! The denoiser pipeline (`denoise.c`): high-pass, windowed analysis FFT,
//! band energies, pitch analysis, feature extraction, the recurrent network,
//! pitch-comb filtering, gain smoothing and windowed overlap-add synthesis,
//! with the one-frame look-ahead delay used by upstream.

use std::sync::{Arc, OnceLock};

use crate::common::{
    biquad, compute_band_corr, compute_band_energy, interp_band_gain, Common,
};
use crate::fft::Cpx;
use crate::nnet::{compute_rnn, RnnState};
use crate::pitch::{pitch_downsample, pitch_search, remove_doubling};
use crate::weights::RnnModel;
use crate::{
    FRAME_SIZE, FREQ_SIZE, NB_BANDS, NB_FEATURES, PITCH_BUF_SIZE, PITCH_FRAME_SIZE,
    PITCH_MAX_PERIOD, PITCH_MIN_PERIOD, WINDOW_SIZE,
};

fn common() -> &'static Common {
    static COMMON: OnceLock<Common> = OnceLock::new();
    COMMON.get_or_init(Common::new)
}

fn default_model() -> Arc<RnnModel> {
    static MODEL: OnceLock<Arc<RnnModel>> = OnceLock::new();
    MODEL.get_or_init(|| Arc::new(RnnModel::default())).clone()
}

/// Streaming RNNoise denoiser. Process audio one 480-sample frame at a time
/// with [`DenoiseState::process_frame`].
pub struct DenoiseState {
    model: Arc<RnnModel>,
    rnn: RnnState,
    analysis_mem: [f32; FRAME_SIZE],
    synthesis_mem: [f32; FRAME_SIZE],
    pitch_buf: [f32; PITCH_BUF_SIZE],
    last_gain: f32,
    last_period: i32,
    mem_hp_x: [f32; 2],
    lastg: [f32; NB_BANDS],
    delayed_x: [Cpx; FREQ_SIZE],
    delayed_p: [Cpx; FREQ_SIZE],
    delayed_ex: [f32; NB_BANDS],
    delayed_ep: [f32; NB_BANDS],
    delayed_exp: [f32; NB_BANDS],
}

impl DenoiseState {
    /// Create a denoiser using the built-in model (shared across instances).
    pub fn new() -> Self {
        Self::with_model(default_model())
    }

    /// Create a denoiser backed by a specific (possibly custom) model.
    pub fn with_model(model: Arc<RnnModel>) -> Self {
        // Build shared tables eagerly so the first frame isn't penalised.
        let _ = common();
        DenoiseState {
            model,
            rnn: RnnState::new(),
            analysis_mem: [0.0; FRAME_SIZE],
            synthesis_mem: [0.0; FRAME_SIZE],
            pitch_buf: [0.0; PITCH_BUF_SIZE],
            last_gain: 0.0,
            last_period: 0,
            mem_hp_x: [0.0; 2],
            lastg: [0.0; NB_BANDS],
            delayed_x: [Cpx::default(); FREQ_SIZE],
            delayed_p: [Cpx::default(); FREQ_SIZE],
            delayed_ex: [0.0; NB_BANDS],
            delayed_ep: [0.0; NB_BANDS],
            delayed_exp: [0.0; NB_BANDS],
        }
    }

    /// Denoise one frame. `input` and `output` must both be [`FRAME_SIZE`]
    /// samples (48 kHz mono, amplitudes in roughly `i16` range). Returns the
    /// voice-activity probability for the frame.
    ///
    /// Note the one-frame algorithmic delay: the output corresponds to the
    /// *previous* input frame.
    pub fn process_frame(&mut self, output: &mut [f32], input: &[f32]) -> f32 {
        assert!(input.len() >= FRAME_SIZE && output.len() >= FRAME_SIZE);

        let mut x = [0.0f32; FRAME_SIZE];
        // DC-removal high-pass.
        const A_HP: [f32; 2] = [-1.99599, 0.99600];
        const B_HP: [f32; 2] = [-2.0, 1.0];
        biquad(&mut x, &mut self.mem_hp_x, &input[..FRAME_SIZE], &B_HP, &A_HP);

        let mut xfreq = [Cpx::default(); FREQ_SIZE];
        let mut p = [Cpx::default(); FREQ_SIZE];
        let mut ex = [0.0f32; NB_BANDS];
        let mut ep = [0.0f32; NB_BANDS];
        let mut exp = [0.0f32; NB_BANDS];
        let mut features = [0.0f32; NB_FEATURES];

        let silence = self.compute_frame_features(
            &mut xfreq, &mut p, &mut ex, &mut ep, &mut exp, &mut features, &x,
        );

        let mut g = [0.0f32; NB_BANDS];
        let mut gf = [0.0f32; FREQ_SIZE];
        gf[0] = 1.0;
        let mut vad_prob = 0.0f32;

        if !silence {
            compute_rnn(
                &self.model,
                &mut self.rnn,
                &mut g,
                std::slice::from_mut(&mut vad_prob),
                &features,
            );
            pitch_filter(
                &mut self.delayed_x,
                &self.delayed_p,
                &self.delayed_ex,
                &self.delayed_ep,
                &self.delayed_exp,
                &g,
            );
            for i in 0..NB_BANDS {
                // Cap decay at 0.6/frame (RT60 ≈ 135 ms) to avoid unnatural attenuation.
                g[i] = g[i].max(0.6 * self.lastg[i]);
                // Compensate for cross-frame energy change.
                let t = g[i] as f64 * (self.delayed_ex[i] as f64 + 1e-3) / (ex[i] as f64 + 1e-3);
                self.lastg[i] = t.min(1.0) as f32;
            }
            interp_band_gain(&mut gf, &g);
            for i in 0..FREQ_SIZE {
                self.delayed_x[i].r *= gf[i];
                self.delayed_x[i].i *= gf[i];
            }
        }

        frame_synthesis(common(), output, &mut self.synthesis_mem, &self.delayed_x);

        self.delayed_x.copy_from_slice(&xfreq);
        self.delayed_p.copy_from_slice(&p);
        self.delayed_ex.copy_from_slice(&ex);
        self.delayed_ep.copy_from_slice(&ep);
        self.delayed_exp.copy_from_slice(&exp);

        vad_prob
    }

    /// `rnn_frame_analysis`: windowed FFT + band energy of the input frame.
    fn frame_analysis(&mut self, xfreq: &mut [Cpx], ex: &mut [f32], input: &[f32]) {
        let c = common();
        let mut x = [0.0f32; WINDOW_SIZE];
        x[..FRAME_SIZE].copy_from_slice(&self.analysis_mem);
        x[FRAME_SIZE..].copy_from_slice(&input[..FRAME_SIZE]);
        self.analysis_mem.copy_from_slice(&input[..FRAME_SIZE]);
        c.apply_window(&mut x);
        c.forward_transform(xfreq, &x);
        compute_band_energy(ex, xfreq);
    }

    /// `rnn_compute_frame_features`: returns `true` if the frame is silence.
    #[allow(clippy::too_many_arguments)]
    fn compute_frame_features(
        &mut self,
        xfreq: &mut [Cpx],
        p: &mut [Cpx],
        ex: &mut [f32],
        ep: &mut [f32],
        exp: &mut [f32],
        features: &mut [f32],
        input: &[f32],
    ) -> bool {
        let c = common();
        self.frame_analysis(xfreq, ex, input);

        self.pitch_buf.copy_within(FRAME_SIZE.., 0);
        self.pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE..].copy_from_slice(&input[..FRAME_SIZE]);

        let mut pitch_ds = [0.0f32; PITCH_BUF_SIZE >> 1];
        pitch_downsample(&self.pitch_buf, &mut pitch_ds, PITCH_BUF_SIZE);

        let coarse = pitch_search(
            &pitch_ds[PITCH_MAX_PERIOD >> 1..],
            &pitch_ds,
            PITCH_FRAME_SIZE,
            PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD,
        );
        let mut pitch_index = PITCH_MAX_PERIOD as i32 - coarse;

        let gain = remove_doubling(
            &pitch_ds,
            PITCH_MAX_PERIOD as i32,
            PITCH_MIN_PERIOD as i32,
            PITCH_FRAME_SIZE as i32,
            &mut pitch_index,
            self.last_period,
            self.last_gain,
        );
        self.last_period = pitch_index;
        self.last_gain = gain;

        let mut pbuf = [0.0f32; WINDOW_SIZE];
        let base = PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index as usize;
        for i in 0..WINDOW_SIZE {
            pbuf[i] = self.pitch_buf[base + i];
        }
        c.apply_window(&mut pbuf);
        c.forward_transform(p, &pbuf);
        compute_band_energy(ep, p);
        compute_band_corr(exp, xfreq, p);
        for i in 0..NB_BANDS {
            let d = 0.001f64 + (ex[i] * ep[i]) as f64;
            exp[i] = (exp[i] as f64 / d.sqrt()) as f32;
        }
        c.dct(&mut features[NB_BANDS..], exp);
        features[2 * NB_BANDS] = 0.01 * (pitch_index as f32 - 300.0);

        let mut ly = [0.0f32; NB_BANDS];
        let mut log_max = -2.0f32;
        let mut follow = -2.0f32;
        let mut e = 0.0f32;
        for i in 0..NB_BANDS {
            let mut lyi = (1e-2f64 + ex[i] as f64).log10() as f32;
            let inner = (follow as f64 - 1.5).max(lyi as f64);
            lyi = ((log_max - 7.0) as f64).max(inner) as f32;
            log_max = log_max.max(lyi);
            follow = (follow as f64 - 1.5).max(lyi as f64) as f32;
            ly[i] = lyi;
            e += ex[i];
        }
        if e < 0.04 {
            for v in features[..NB_FEATURES].iter_mut() {
                *v = 0.0;
            }
            return true;
        }
        c.dct(&mut features[..NB_BANDS], &ly);
        features[0] -= 12.0;
        features[1] -= 4.0;
        false
    }
}

impl Default for DenoiseState {
    fn default() -> Self {
        Self::new()
    }
}

/// `frame_synthesis`: inverse transform, window, overlap-add with the stored
/// synthesis memory.
fn frame_synthesis(c: &Common, out: &mut [f32], synthesis_mem: &mut [f32], y: &[Cpx]) {
    let mut x = [0.0f32; WINDOW_SIZE];
    c.inverse_transform(&mut x, y);
    c.apply_window(&mut x);
    for i in 0..FRAME_SIZE {
        out[i] = x[i] + synthesis_mem[i];
    }
    synthesis_mem.copy_from_slice(&x[FRAME_SIZE..]);
}

/// `rnn_pitch_filter`: comb-filter the spectrum towards the pitch-predicted
/// spectrum `p`, then renormalise band energy.
fn pitch_filter(
    xfreq: &mut [Cpx],
    p: &[Cpx],
    ex: &[f32],
    ep: &[f32],
    exp: &[f32],
    g: &[f32],
) {
    let mut r = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS {
        let mut ri = if exp[i] > g[i] {
            1.0f32
        } else {
            let sq_exp = exp[i] * exp[i];
            let sq_g = g[i] * g[i];
            let num = sq_exp * (1.0 - sq_g);
            let den = 0.001f64 + (sq_g * (1.0 - sq_exp)) as f64;
            (num as f64 / den) as f32
        };
        ri = (ri.min(1.0).max(0.0) as f64).sqrt() as f32;
        ri = (ri as f64 * (ex[i] as f64 / (1e-8 + ep[i] as f64)).sqrt()) as f32;
        r[i] = ri;
    }
    let mut rf = [0.0f32; FREQ_SIZE];
    interp_band_gain(&mut rf, &r);
    for i in 0..FREQ_SIZE {
        xfreq[i].r += rf[i] * p[i].r;
        xfreq[i].i += rf[i] * p[i].i;
    }
    let mut new_e = [0.0f32; NB_BANDS];
    compute_band_energy(&mut new_e, xfreq);
    let mut norm = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS {
        norm[i] = (ex[i] as f64 / (1e-8 + new_e[i] as f64)).sqrt() as f32;
    }
    let mut normf = [0.0f32; FREQ_SIZE];
    interp_band_gain(&mut normf, &norm);
    for i in 0..FREQ_SIZE {
        xfreq[i].r *= normf[i];
        xfreq[i].i *= normf[i];
    }
}
