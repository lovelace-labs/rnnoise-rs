//! Optional support for the **original** (2018) RNNoise model — the 22-band,
//! ~215 K-parameter dense + 3-GRU network that
//! [`nnnoiseless`](https://github.com/jneem/nnnoiseless) also implements.
//! Enabled with the `legacy-model` feature.
//!
//! This is a separate, smaller/faster (but lower-quality) pipeline from the
//! current model in [`crate::denoise`]. The DSP front-end shares this crate's
//! FFT, pitch and biquad code; the bands, features, network and weight format
//! all differ. Output matches the old model to within ~1 LSB (the residual is
//! the KISS-FFT-vs-rustfft difference, the same spread `nnnoiseless` has vs the
//! original C).

use std::sync::{Arc, OnceLock};

use crate::common::biquad;
use crate::fft::{Cpx, RealFft};
use crate::pitch::{pitch_downsample, pitch_search, remove_doubling};
use crate::weights::ModelError;
use crate::{
    FRAME_SIZE, FREQ_SIZE, PITCH_BUF_SIZE, PITCH_FRAME_SIZE, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
    WINDOW_SIZE,
};

const NB_BANDS: usize = 22;
const CEPS_MEM: usize = 8;
const NB_DELTA_CEPS: usize = 6;
const NB_FEATURES: usize = NB_BANDS + 3 * NB_DELTA_CEPS + 2; // 42
const FRAME_SIZE_SHIFT: usize = 2;
const WEIGHTS_SCALE: f32 = 1.0 / 256.0;
const MAX_NEURONS: usize = 128;

/// Band edges (in 5 ms-spaced units; multiply by `1<<FRAME_SIZE_SHIFT` for bins).
const EBAND_5MS: [usize; NB_BANDS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

#[cfg(feature = "legacy-model")]
static LEGACY_BLOB: &[u8] = include_bytes!("../models/rnnoise_legacy.rnn");

// --- shared tables (window, dct, fft, wnorm) -------------------------------

struct LegacyCommon {
    window: [f32; WINDOW_SIZE],
    dct_table: [f32; NB_BANDS * NB_BANDS],
    wnorm: f32,
    rfft: RealFft,
}

const DCT_SCALE: f64 = 0.301_511_344_577_763_5; // sqrt(2/22), as in upstream

impl LegacyCommon {
    fn new() -> Self {
        use std::f64::consts::PI;
        let mut window = [0.0f32; WINDOW_SIZE];
        for i in 0..FRAME_SIZE {
            let inner = (0.5 * PI * (i as f64 + 0.5) / FRAME_SIZE as f64).sin();
            let w = (0.5 * PI * inner * inner).sin() as f32;
            window[i] = w;
            window[WINDOW_SIZE - 1 - i] = w;
        }
        let wnorm = 1.0 / window.iter().map(|w| w * w).sum::<f32>();

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
        LegacyCommon {
            window,
            dct_table,
            wnorm,
            rfft: RealFft::new(WINDOW_SIZE),
        }
    }

    fn apply_window(&self, x: &mut [f32]) {
        for (v, &w) in x.iter_mut().zip(self.window.iter()) {
            *v *= w;
        }
    }

    /// Windowed real signal → spectrum (real-input FFT), normalised by `wnorm`.
    fn forward(&self, input: &[f32], out: &mut [Cpx; FREQ_SIZE]) {
        self.rfft.forward(input, out);
        let w = self.wnorm;
        for c in out.iter_mut() {
            c.r *= w;
            c.i *= w;
        }
    }

    /// Spectrum → windowed real signal (real inverse FFT, halved as upstream).
    fn inverse(&self, x: &[Cpx; FREQ_SIZE], out: &mut [f32; WINDOW_SIZE]) {
        self.rfft.inverse(x, out);
        for v in out.iter_mut() {
            *v *= 0.5;
        }
    }

    fn dct(&self, out: &mut [f32], input: &[f32]) {
        for i in 0..NB_BANDS {
            let mut sum = 0.0f32;
            for j in 0..NB_BANDS {
                sum += input[j] * self.dct_table[j * NB_BANDS + i];
            }
            out[i] = (sum as f64 * DCT_SCALE) as f32;
        }
    }
}

fn legacy_common() -> &'static LegacyCommon {
    static C: OnceLock<LegacyCommon> = OnceLock::new();
    C.get_or_init(LegacyCommon::new)
}

/// `compute_band_corr` (old 22-band variant; band energy is `corr(x, x)`).
fn compute_band_corr(out: &mut [f32], x: &[Cpx], p: &[Cpx]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND_5MS[i + 1] - EBAND_5MS[i]) << FRAME_SIZE_SHIFT;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = (EBAND_5MS[i] << FRAME_SIZE_SHIFT) + j;
            let corr = x[idx].r * p[idx].r + x[idx].i * p[idx].i;
            out[i] += (1.0 - frac) * corr;
            out[i + 1] += frac * corr;
        }
    }
    out[0] *= 2.0;
    out[NB_BANDS - 1] *= 2.0;
}

/// Copy the `lag`-shifted `WINDOW_SIZE` tail of `input_mem` and window it.
fn windowed(lc: &LegacyCommon, input_mem: &[f32], lag: usize) -> [f32; WINDOW_SIZE] {
    let start = PITCH_BUF_SIZE - WINDOW_SIZE - lag;
    let mut buf = [0.0f32; WINDOW_SIZE];
    buf.copy_from_slice(&input_mem[start..start + WINDOW_SIZE]);
    lc.apply_window(&mut buf);
    buf
}

fn interp_band_gain(g: &mut [f32], band_e: &[f32]) {
    for v in g.iter_mut() {
        *v = 0.0;
    }
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND_5MS[i + 1] - EBAND_5MS[i]) << FRAME_SIZE_SHIFT;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = (EBAND_5MS[i] << FRAME_SIZE_SHIFT) + j;
            g[idx] = (1.0 - frac) * band_e[i] + frac * band_e[i + 1];
        }
    }
}

// --- activations (table-based, like the old model) -------------------------

const TANSIG_TABLE: [f32; 201] = [
    0.000000, 0.039979, 0.079830, 0.119427, 0.158649, 0.197375, 0.235496, 0.272905, 0.309507,
    0.345214, 0.379949, 0.413644, 0.446244, 0.477700, 0.507977, 0.537050, 0.564900, 0.591519,
    0.616909, 0.641077, 0.664037, 0.685809, 0.706419, 0.725897, 0.744277, 0.761594, 0.777888,
    0.793199, 0.807569, 0.821040, 0.833655, 0.845456, 0.856485, 0.866784, 0.876393, 0.885352,
    0.893698, 0.901468, 0.908698, 0.915420, 0.921669, 0.927473, 0.932862, 0.937863, 0.942503,
    0.946806, 0.950795, 0.954492, 0.957917, 0.961090, 0.964028, 0.966747, 0.969265, 0.971594,
    0.973749, 0.975743, 0.977587, 0.979293, 0.980869, 0.982327, 0.983675, 0.984921, 0.986072,
    0.987136, 0.988119, 0.989027, 0.989867, 0.990642, 0.991359, 0.992020, 0.992631, 0.993196,
    0.993718, 0.994199, 0.994644, 0.995055, 0.995434, 0.995784, 0.996108, 0.996407, 0.996682,
    0.996937, 0.997172, 0.997389, 0.997590, 0.997775, 0.997946, 0.998104, 0.998249, 0.998384,
    0.998508, 0.998623, 0.998728, 0.998826, 0.998916, 0.999000, 0.999076, 0.999147, 0.999213,
    0.999273, 0.999329, 0.999381, 0.999428, 0.999472, 0.999513, 0.999550, 0.999585, 0.999617,
    0.999646, 0.999673, 0.999699, 0.999722, 0.999743, 0.999763, 0.999781, 0.999798, 0.999813,
    0.999828, 0.999841, 0.999853, 0.999865, 0.999875, 0.999885, 0.999893, 0.999902, 0.999909,
    0.999916, 0.999923, 0.999929, 0.999934, 0.999939, 0.999944, 0.999948, 0.999952, 0.999956,
    0.999959, 0.999962, 0.999965, 0.999968, 0.999970, 0.999973, 0.999975, 0.999977, 0.999978,
    0.999980, 0.999982, 0.999983, 0.999984, 0.999986, 0.999987, 0.999988, 0.999989, 0.999990,
    0.999990, 0.999991, 0.999992, 0.999992, 0.999993, 0.999994, 0.999994, 0.999994, 0.999995,
    0.999995, 0.999996, 0.999996, 0.999996, 0.999997, 0.999997, 0.999997, 0.999997, 0.999997,
    0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999998, 0.999999, 0.999999, 0.999999,
    0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999, 0.999999,
    0.999999, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000,
    1.000000, 1.000000, 1.000000,
];

#[allow(clippy::neg_cmp_op_on_partial_ord)] // reversed comparisons catch NaN (upstream behaviour)
fn tansig_approx(x: f32) -> f32 {
    if !(x < 8.0) {
        return 1.0;
    }
    if !(x > -8.0) {
        return -1.0;
    }
    let (mut x, sign) = if x < 0.0 { (-x, -1.0) } else { (x, 1.0) };
    let i = (0.5 + 25.0 * x).floor();
    x -= 0.04 * i;
    let y = TANSIG_TABLE[i as usize];
    let dy = 1.0 - y * y;
    let y = y + x * dy * (1.0 - y * x);
    sign * y
}

fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tansig_approx(0.5 * x)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Act {
    Tanh,
    Sigmoid,
    Relu,
}

impl Act {
    fn from_i8(x: i8) -> Option<Act> {
        match x {
            0 => Some(Act::Tanh),
            1 => Some(Act::Sigmoid),
            2 => Some(Act::Relu),
            _ => None,
        }
    }
    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            Act::Tanh => tansig_approx(x),
            Act::Sigmoid => sigmoid_approx(x),
            Act::Relu => x.max(0.0),
        }
    }
}

// --- network ---------------------------------------------------------------

struct DenseLayer {
    bias: Vec<i8>,
    weights: Vec<i8>, // [nb_inputs][nb_neurons]
    nb_inputs: usize,
    nb_neurons: usize,
    activation: Act,
}

impl DenseLayer {
    fn compute(&self, out: &mut [f32], input: &[f32]) {
        let n = self.nb_neurons;
        for i in 0..n {
            out[i] = self.bias[i] as f32;
        }
        for (col, &inp) in self.weights.chunks_exact(n).zip(&input[..self.nb_inputs]) {
            for (o, &w) in out[..n].iter_mut().zip(col) {
                *o += w as f32 * inp;
            }
        }
        for o in out[..n].iter_mut() {
            *o = self.activation.apply(*o * WEIGHTS_SCALE);
        }
    }
}

struct GruLayer {
    bias: Vec<i8>,              // [3*nb_neurons]
    input_weights: Vec<i8>,     // [nb_inputs][3*nb_neurons]
    recurrent_weights: Vec<i8>, // [nb_neurons][3*nb_neurons]
    nb_inputs: usize,
    nb_neurons: usize,
    activation: Act,
}

/// `out[k] += sum_c data[c][offset+k] * input[c]`, where each column has length
/// `stride` (= 3*nb_neurons).
fn gru_mul_add(out: &mut [f32], data: &[i8], stride: usize, offset: usize, input: &[f32]) {
    let n = out.len();
    for (col, &inp) in data.chunks_exact(stride).zip(input) {
        for (o, &w) in out.iter_mut().zip(&col[offset..offset + n]) {
            *o += w as f32 * inp;
        }
    }
}

impl GruLayer {
    fn compute(&self, state: &mut [f32], input: &[f32]) {
        let n = self.nb_neurons;
        let mut z = [0.0f32; MAX_NEURONS];
        let mut r = [0.0f32; MAX_NEURONS];
        let mut h = [0.0f32; MAX_NEURONS];

        // Update gate z.
        for i in 0..n {
            z[i] = self.bias[i] as f32;
        }
        gru_mul_add(
            &mut z[..n],
            &self.input_weights,
            3 * n,
            0,
            &input[..self.nb_inputs],
        );
        gru_mul_add(&mut z[..n], &self.recurrent_weights, 3 * n, 0, &state[..n]);
        for zi in z[..n].iter_mut() {
            *zi = sigmoid_approx(*zi * WEIGHTS_SCALE);
        }

        // Reset gate r, pre-multiplied by the state.
        for i in 0..n {
            r[i] = self.bias[n + i] as f32;
        }
        gru_mul_add(
            &mut r[..n],
            &self.input_weights,
            3 * n,
            n,
            &input[..self.nb_inputs],
        );
        gru_mul_add(&mut r[..n], &self.recurrent_weights, 3 * n, n, &state[..n]);
        for (ri, &s) in r[..n].iter_mut().zip(&state[..n]) {
            *ri = s * sigmoid_approx(*ri * WEIGHTS_SCALE);
        }

        // Candidate h.
        for i in 0..n {
            h[i] = self.bias[2 * n + i] as f32;
        }
        gru_mul_add(
            &mut h[..n],
            &self.input_weights,
            3 * n,
            2 * n,
            &input[..self.nb_inputs],
        );
        gru_mul_add(&mut h[..n], &self.recurrent_weights, 3 * n, 2 * n, &r[..n]);
        for i in 0..n {
            let ha = self.activation.apply(h[i] * WEIGHTS_SCALE);
            state[i] = z[i] * state[i] + (1.0 - z[i]) * ha;
        }
    }
}

/// The original RNNoise model (dense + 3 GRUs + 2 output heads).
pub struct RnnModelV1 {
    input_dense: DenseLayer,
    vad_gru: GruLayer,
    noise_gru: GruLayer,
    denoise_gru: GruLayer,
    denoise_output: DenseLayer,
    vad_output: DenseLayer,
}

struct Reader<'a> {
    data: &'a [i8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u(&mut self) -> Result<usize, ModelError> {
        let b = *self.data.get(self.pos).ok_or(ModelError::Malformed)?;
        self.pos += 1;
        if b < 0 {
            return Err(ModelError::Malformed);
        }
        Ok(b as usize)
    }
    fn take(&mut self, n: usize) -> Result<Vec<i8>, ModelError> {
        let end = self.pos.checked_add(n).ok_or(ModelError::Malformed)?;
        let s = self.data.get(self.pos..end).ok_or(ModelError::Malformed)?;
        self.pos = end;
        Ok(s.to_vec())
    }
    fn act(&mut self) -> Result<Act, ModelError> {
        let b = *self.data.get(self.pos).ok_or(ModelError::Malformed)?;
        self.pos += 1;
        Act::from_i8(b).ok_or(ModelError::Malformed)
    }
    fn dense(&mut self) -> Result<DenseLayer, ModelError> {
        let nb_inputs = self.u()?;
        let nb_neurons = self.u()?;
        let activation = self.act()?;
        let weights = self.take(nb_neurons * nb_inputs)?;
        let bias = self.take(nb_neurons)?;
        Ok(DenseLayer {
            bias,
            weights,
            nb_inputs,
            nb_neurons,
            activation,
        })
    }
    fn gru(&mut self) -> Result<GruLayer, ModelError> {
        let nb_inputs = self.u()?;
        let nb_neurons = self.u()?;
        let activation = self.act()?;
        let input_weights = self.take(3 * nb_neurons * nb_inputs)?;
        let recurrent_weights = self.take(3 * nb_neurons * nb_neurons)?;
        let bias = self.take(3 * nb_neurons)?;
        Ok(GruLayer {
            bias,
            input_weights,
            recurrent_weights,
            nb_inputs,
            nb_neurons,
            activation,
        })
    }
}

impl RnnModelV1 {
    /// Parse a model in the `nnnoiseless` weight format (concatenated `i8`
    /// layer records: input-dense, vad-gru, noise-gru, denoise-gru,
    /// denoise-output, vad-output).
    pub fn from_bytes(bytes: &[u8]) -> Result<RnnModelV1, ModelError> {
        // SAFETY-free reinterpretation of u8 as i8 (same size/layout).
        let data: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        let mut r = Reader {
            data: &data,
            pos: 0,
        };
        let input_dense = r.dense()?;
        let vad_gru = r.gru()?;
        let noise_gru = r.gru()?;
        let denoise_gru = r.gru()?;
        let denoise_output = r.dense()?;
        let vad_output = r.dense()?;
        if r.pos != data.len() {
            return Err(ModelError::Malformed);
        }
        if input_dense.nb_inputs != NB_FEATURES
            || denoise_output.nb_neurons != NB_BANDS
            || vad_output.nb_neurons != 1
            || input_dense.nb_neurons != vad_gru.nb_inputs
            || NB_FEATURES + input_dense.nb_neurons + vad_gru.nb_neurons != noise_gru.nb_inputs
            || NB_FEATURES + vad_gru.nb_neurons + noise_gru.nb_neurons != denoise_gru.nb_inputs
            || denoise_gru.nb_neurons != denoise_output.nb_inputs
            || vad_gru.nb_neurons != vad_output.nb_inputs
            || noise_gru.nb_neurons > MAX_NEURONS
            || denoise_gru.nb_neurons > MAX_NEURONS
        {
            return Err(ModelError::Malformed);
        }
        Ok(RnnModelV1 {
            input_dense,
            vad_gru,
            noise_gru,
            denoise_gru,
            denoise_output,
            vad_output,
        })
    }
}

#[cfg(feature = "legacy-model")]
impl Default for RnnModelV1 {
    fn default() -> Self {
        RnnModelV1::from_bytes(LEGACY_BLOB).expect("embedded legacy model is valid")
    }
}

struct RnnStateV1 {
    vad_gru_state: Vec<f32>,
    noise_gru_state: Vec<f32>,
    denoise_gru_state: Vec<f32>,
}

impl RnnStateV1 {
    fn new(m: &RnnModelV1) -> Self {
        RnnStateV1 {
            vad_gru_state: vec![0.0; m.vad_gru.nb_neurons],
            noise_gru_state: vec![0.0; m.noise_gru.nb_neurons],
            denoise_gru_state: vec![0.0; m.denoise_gru.nb_neurons],
        }
    }

    fn compute(&mut self, m: &RnnModelV1, gains: &mut [f32], vad: &mut [f32], features: &[f32]) {
        let dn = m.input_dense.nb_neurons;
        let vn = m.vad_gru.nb_neurons;
        let nn = m.noise_gru.nb_neurons;
        let mut dense_out = [0.0f32; MAX_NEURONS];
        m.input_dense.compute(&mut dense_out, features);
        m.vad_gru.compute(&mut self.vad_gru_state, &dense_out[..dn]);
        m.vad_output.compute(vad, &self.vad_gru_state);

        // noise_gru input = [dense_out, vad_gru_state, features]
        let mut noise_in = [0.0f32; MAX_NEURONS * 2 + NB_FEATURES];
        noise_in[..dn].copy_from_slice(&dense_out[..dn]);
        noise_in[dn..dn + vn].copy_from_slice(&self.vad_gru_state);
        noise_in[dn + vn..dn + vn + NB_FEATURES].copy_from_slice(features);
        m.noise_gru.compute(&mut self.noise_gru_state, &noise_in);

        // denoise_gru input = [vad_gru_state, noise_gru_state, features]
        let mut denoise_in = [0.0f32; MAX_NEURONS * 2 + NB_FEATURES];
        denoise_in[..vn].copy_from_slice(&self.vad_gru_state);
        denoise_in[vn..vn + nn].copy_from_slice(&self.noise_gru_state);
        denoise_in[vn + nn..vn + nn + NB_FEATURES].copy_from_slice(features);
        m.denoise_gru
            .compute(&mut self.denoise_gru_state, &denoise_in);

        m.denoise_output.compute(gains, &self.denoise_gru_state);
    }
}

// --- the denoiser ----------------------------------------------------------

/// Streaming denoiser using the **original** RNNoise model. Same usage as
/// [`crate::DenoiseState`] (480-sample frames, 48 kHz mono), but smaller and
/// faster at lower quality.
pub struct DenoiseStateV1 {
    model: Arc<RnnModelV1>,
    rnn: RnnStateV1,
    lastg: [f32; NB_BANDS],
    input_mem: [f32; PITCH_BUF_SIZE],
    cepstral_mem: [[f32; NB_BANDS]; CEPS_MEM],
    mem_id: usize,
    mem_hp_x: [f32; 2],
    synthesis_mem: [f32; FRAME_SIZE],
    last_period: i32,
    last_gain: f32,
    x: [Cpx; FREQ_SIZE],
    p: [Cpx; FREQ_SIZE],
    ex: [f32; NB_BANDS],
    ep: [f32; NB_BANDS],
    exp: [f32; NB_BANDS],
    features: [f32; NB_FEATURES],
}

impl DenoiseStateV1 {
    /// Create a denoiser using a (shared) legacy model.
    pub fn with_model(model: Arc<RnnModelV1>) -> Self {
        let _ = legacy_common();
        let rnn = RnnStateV1::new(&model);
        DenoiseStateV1 {
            model,
            rnn,
            lastg: [0.0; NB_BANDS],
            input_mem: [0.0; PITCH_BUF_SIZE],
            cepstral_mem: [[0.0; NB_BANDS]; CEPS_MEM],
            mem_id: 0,
            mem_hp_x: [0.0; 2],
            synthesis_mem: [0.0; FRAME_SIZE],
            last_period: 0,
            last_gain: 0.0,
            x: [Cpx::default(); FREQ_SIZE],
            p: [Cpx::default(); FREQ_SIZE],
            ex: [0.0; NB_BANDS],
            ep: [0.0; NB_BANDS],
            exp: [0.0; NB_BANDS],
            features: [0.0; NB_FEATURES],
        }
    }

    /// Create a denoiser using the built-in legacy model.
    #[cfg(feature = "legacy-model")]
    pub fn new() -> Self {
        static MODEL: OnceLock<Arc<RnnModelV1>> = OnceLock::new();
        let m = MODEL
            .get_or_init(|| Arc::new(RnnModelV1::default()))
            .clone();
        Self::with_model(m)
    }

    /// Denoise one [`FRAME_SIZE`]-sample frame; returns the VAD probability.
    pub fn process_frame(&mut self, output: &mut [f32], input: &[f32]) -> f32 {
        assert!(input.len() >= FRAME_SIZE && output.len() >= FRAME_SIZE);
        let lc = legacy_common();

        // Shift input history and high-pass filter the new frame into it.
        let new_idx = PITCH_BUF_SIZE - FRAME_SIZE;
        self.input_mem.copy_within(FRAME_SIZE.., 0);
        let mut filt = [0.0f32; FRAME_SIZE];
        biquad(
            &mut filt,
            &mut self.mem_hp_x,
            &input[..FRAME_SIZE],
            &[-2.0, 1.0],
            &[-1.99599, 0.99600],
        );
        self.input_mem[new_idx..].copy_from_slice(&filt);

        let silence = self.compute_frame_features(lc);

        let mut g = [0.0f32; NB_BANDS];
        let mut gf = [1.0f32; FREQ_SIZE];
        let mut vad = [0.0f32];
        if !silence {
            let model = self.model.clone();
            self.rnn.compute(&model, &mut g, &mut vad, &self.features);
            self.pitch_filter(&g);
            for i in 0..NB_BANDS {
                g[i] = g[i].max(0.6 * self.lastg[i]);
                self.lastg[i] = g[i];
            }
            interp_band_gain(&mut gf, &g);
            for (xi, &gfi) in self.x.iter_mut().zip(gf.iter()) {
                xi.r *= gfi;
                xi.i *= gfi;
            }
        }
        self.frame_synthesis(lc, output);
        vad[0]
    }

    fn find_pitch(&mut self) -> usize {
        let mut ds = [0.0f32; PITCH_BUF_SIZE / 2];
        pitch_downsample(&self.input_mem, &mut ds, PITCH_BUF_SIZE);
        let coarse = pitch_search(
            &ds[PITCH_MAX_PERIOD / 2..],
            &ds,
            PITCH_FRAME_SIZE,
            PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD,
        );
        let mut t0 = PITCH_MAX_PERIOD as i32 - coarse;
        let gain = remove_doubling(
            &ds,
            PITCH_MAX_PERIOD as i32,
            PITCH_MIN_PERIOD as i32,
            PITCH_FRAME_SIZE as i32,
            &mut t0,
            self.last_period,
            self.last_gain,
        );
        self.last_period = t0;
        self.last_gain = gain;
        t0 as usize
    }

    fn compute_frame_features(&mut self, lc: &LegacyCommon) -> bool {
        let mut ly = [0.0f32; NB_BANDS];
        let mut tmp = [0.0f32; NB_BANDS];

        // Pitch is found in the time domain (independent of the FFTs), so we can
        // window both the signal and the pitch-lagged signal first and transform
        // them together in a single complex FFT.
        let pitch_idx = self.find_pitch();
        let xbuf = windowed(lc, &self.input_mem, 0);
        let pbuf = windowed(lc, &self.input_mem, pitch_idx);
        lc.forward(&xbuf, &mut self.x);
        lc.forward(&pbuf, &mut self.p);
        compute_band_corr(&mut self.ex, &self.x, &self.x);
        compute_band_corr(&mut self.ep, &self.p, &self.p);

        compute_band_corr(&mut self.exp, &self.x, &self.p);
        for i in 0..NB_BANDS {
            self.exp[i] /= (0.001 + self.ex[i] * self.ep[i]).sqrt();
        }
        lc.dct(&mut tmp, &self.exp);
        for i in 0..NB_DELTA_CEPS {
            self.features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = tmp[i];
        }
        self.features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3;
        self.features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9;
        self.features[NB_BANDS + 3 * NB_DELTA_CEPS] = 0.01 * (pitch_idx as f32 - 300.0);

        let mut log_max = -2.0f32;
        let mut follow = -2.0f32;
        let mut e = 0.0f32;
        for i in 0..NB_BANDS {
            ly[i] = (1e-2 + self.ex[i])
                .log10()
                .max(log_max - 7.0)
                .max(follow - 1.5);
            log_max = log_max.max(ly[i]);
            follow = (follow - 1.5).max(ly[i]);
            e += self.ex[i];
        }
        if e < 0.04 {
            for v in self.features.iter_mut() {
                *v = 0.0;
            }
            return true;
        }
        lc.dct(&mut self.features, &ly);
        self.features[0] -= 12.0;
        self.features[1] -= 4.0;

        let ceps_0 = self.mem_id;
        let ceps_1 = if self.mem_id < 1 {
            CEPS_MEM + self.mem_id - 1
        } else {
            self.mem_id - 1
        };
        let ceps_2 = if self.mem_id < 2 {
            CEPS_MEM + self.mem_id - 2
        } else {
            self.mem_id - 2
        };
        for i in 0..NB_BANDS {
            self.cepstral_mem[ceps_0][i] = self.features[i];
        }
        self.mem_id += 1;
        for i in 0..NB_DELTA_CEPS {
            let c0 = self.cepstral_mem[ceps_0][i];
            let c1 = self.cepstral_mem[ceps_1][i];
            let c2 = self.cepstral_mem[ceps_2][i];
            self.features[i] = c0 + c1 + c2;
            self.features[NB_BANDS + i] = c0 - c2;
            self.features[NB_BANDS + NB_DELTA_CEPS + i] = c0 - 2.0 * c1 + c2;
        }

        let mut spec_variability = 0.0f32;
        if self.mem_id == CEPS_MEM {
            self.mem_id = 0;
        }
        for i in 0..CEPS_MEM {
            let mut min_dist = 1e15f32;
            for j in 0..CEPS_MEM {
                let mut dist = 0.0f32;
                for k in 0..NB_BANDS {
                    let d = self.cepstral_mem[i][k] - self.cepstral_mem[j][k];
                    dist += d * d;
                }
                if j != i {
                    min_dist = min_dist.min(dist);
                }
            }
            spec_variability += min_dist;
        }
        self.features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM as f32 - 2.1;
        false
    }

    fn pitch_filter(&mut self, gain: &[f32; NB_BANDS]) {
        let mut r = [0.0f32; NB_BANDS];
        for i in 0..NB_BANDS {
            r[i] = if self.exp[i] > gain[i] {
                1.0
            } else {
                let exp_sq = self.exp[i] * self.exp[i];
                let g_sq = gain[i] * gain[i];
                exp_sq * (1.0 - g_sq) / (0.001 + g_sq * (1.0 - exp_sq))
            };
            r[i] = r[i].clamp(0.0, 1.0).sqrt();
            r[i] *= (self.ex[i] / (1e-8 + self.ep[i])).sqrt();
        }
        let mut rf = [0.0f32; FREQ_SIZE];
        interp_band_gain(&mut rf, &r);
        for ((xi, pi), &rfi) in self.x.iter_mut().zip(self.p.iter()).zip(rf.iter()) {
            xi.r += rfi * pi.r;
            xi.i += rfi * pi.i;
        }
        let mut new_e = [0.0f32; NB_BANDS];
        compute_band_corr(&mut new_e, &self.x, &self.x);
        for i in 0..NB_BANDS {
            r[i] = (self.ex[i] / (1e-8 + new_e[i])).sqrt();
        }
        interp_band_gain(&mut rf, &r);
        for (xi, &rfi) in self.x.iter_mut().zip(rf.iter()) {
            xi.r *= rfi;
            xi.i *= rfi;
        }
    }

    fn frame_synthesis(&mut self, lc: &LegacyCommon, out: &mut [f32]) {
        let mut buf = [0.0f32; WINDOW_SIZE];
        lc.inverse(&self.x, &mut buf);
        lc.apply_window(&mut buf);
        for i in 0..FRAME_SIZE {
            out[i] = buf[i] + self.synthesis_mem[i];
            self.synthesis_mem[i] = buf[FRAME_SIZE + i];
        }
    }
}

#[cfg(feature = "legacy-model")]
impl Default for DenoiseStateV1 {
    fn default() -> Self {
        Self::new()
    }
}
