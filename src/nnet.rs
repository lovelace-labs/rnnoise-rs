//! Neural-network primitives ported from `nnet.c` / `nnet_arch.h` / `vec.h`
//! (scalar `_c` path, float weights). Covers the dense/sparse affine transform,
//! Conv1D, GRU, and the tanh/sigmoid rational-polynomial approximations.

/// Activation kinds used by the RNNoise model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Activation {
    Tanh,
    Sigmoid,
    /// Identity — included for completeness with the upstream layer model.
    #[allow(dead_code)]
    Linear,
}

// vec.h `tanh_approx` rational-polynomial coefficients.
const N0: f32 = 952.528_015_14;
const N1: f32 = 96.392_356_87;
const N2: f32 = 0.608_630_42;
const D0: f32 = 952.723_999_02;
const D1: f32 = 413.368_011_47;
const D2: f32 = 11.886_009_22;

#[inline(always)]
pub(crate) fn tanh_approx(x: f32) -> f32 {
    let x2 = x * x;
    let num = (N2 * x2 + N1) * x2 + N0;
    let den = (D2 * x2 + D1) * x2 + D0;
    let num = num * x / den;
    num.min(1.0).max(-1.0)
}

#[inline(always)]
pub(crate) fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tanh_approx(0.5 * x)
}

#[inline]
fn compute_activation(out: &mut [f32], act: Activation) {
    match act {
        Activation::Tanh => {
            for v in out.iter_mut() {
                *v = tanh_approx(*v);
            }
        }
        Activation::Sigmoid => {
            for v in out.iter_mut() {
                *v = sigmoid_approx(*v);
            }
        }
        Activation::Linear => {}
    }
}

/// Block size for the sparse 8×4 GEMV layout (`SPARSE_BLOCK_SIZE` in vec.h is 32,
/// i.e. 8 rows × 4 columns of weights per block).
const SPARSE_ROWS: usize = 8;

/// Largest Conv1D input width in the model (conv2: 128×3 = 384).
const MAX_CONV_INPUTS: usize = 384;
/// Largest GRU `3*n` in the model (n = 384).
const MAX_GRU_3N: usize = 1152;

/// Weight storage for a layer: full-precision float (bit-exact default) or an
/// opt-in int8 quantization. The int8 form is a **transposed, dense**
/// `[output][input]` matrix with a per-output scale, which lets each output be a
/// contiguous int8 dot product — ideal for the NEON `sdot` path.
pub(crate) enum Weights {
    Float(Vec<f32>),
    Q8 { w: Vec<i8>, scale: Vec<f32> },
}

/// Generic (sparse) affine transform — port of `LinearLayer` + `compute_linear`.
/// Always carries a bias for the RNNoise model.
pub(crate) struct LinearLayer {
    pub bias: Vec<f32>,
    pub weights: Weights,
    /// Present for sparse layers (the GRU input/recurrent matrices).
    pub weights_idx: Option<Vec<i32>>,
    /// Present for GRU recurrent matrices.
    pub diag: Option<Vec<f32>>,
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}

impl LinearLayer {
    /// `compute_linear`: `out = W·in + bias (+ diag·in for GRU)`.
    fn compute(&self, out: &mut [f32], input: &[f32]) {
        let n = self.nb_outputs;
        let m = self.nb_inputs;
        let out = &mut out[..n];
        match (&self.weights, &self.weights_idx) {
            (Weights::Float(w), Some(idx)) => sparse_sgemv8x4(out, w, idx, n, input),
            (Weights::Float(w), None) => dense_sgemv(out, w, m, n, input),
            (Weights::Q8 { w, scale }, _) => dense_q8(out, w, scale, m, n, input),
        }
        for i in 0..n {
            out[i] += self.bias[i];
        }
        if let Some(diag) = &self.diag {
            // diag only used for GRU recurrent weights: 3*M == N.
            debug_assert_eq!(3 * m, n);
            for i in 0..m {
                out[i] += diag[i] * input[i];
                out[i + m] += diag[i + m] * input[i];
                out[i + 2 * m] += diag[i + 2 * m] * input[i];
            }
        }
    }

    /// Replace float weights with an int8 quantization. No-op if already
    /// quantized. Used by [`crate::RnnModel::quantized`].
    pub(crate) fn quantize(&mut self) {
        let Weights::Float(w) = &self.weights else {
            return;
        };
        let (n, m) = (self.nb_outputs, self.nb_inputs);

        // Reconstruct a dense `[output][input]` float matrix. The shipped model
        // is dense whether stored row-major (`[input][output]`) or in sparse
        // 8×4 blocks, so this just transposes / un-blocks it.
        let mut dense = vec![0.0f32; n * m];
        match &self.weights_idx {
            None => {
                for o in 0..n {
                    for j in 0..m {
                        dense[o * m + j] = w[j * n + o];
                    }
                }
            }
            Some(idx) => {
                for_each_block(idx, n, |base, pos, wi| {
                    for c in 0..4 {
                        for r in 0..SPARSE_ROWS {
                            dense[(base + r) * m + (pos + c)] = w[wi + c * 8 + r];
                        }
                    }
                });
            }
        }

        // Per-output symmetric quantization. The scale folds in *both* the
        // weight scale (max/127) and the activation scale (1/127), so the
        // runtime is `out[o] = scale[o] * Σ q_w·q_x` with an int32 dot product.
        let mut scale = vec![0.0f32; n];
        let mut q = vec![0i8; n * m];
        for o in 0..n {
            let row = &dense[o * m..o * m + m];
            let mx = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            scale[o] = mx / (127.0 * 127.0);
            if mx > 0.0 {
                let inv = 127.0 / mx;
                for (qi, &v) in q[o * m..o * m + m].iter_mut().zip(row) {
                    *qi = quant(v * inv);
                }
            }
        }
        self.weights = Weights::Q8 { w: q, scale };
    }
}

#[inline]
fn quant(x: f32) -> i8 {
    (x.round() as i32).clamp(-127, 127) as i8
}

/// Quantize an activation to int8: `floor(0.5 + 127·x)` clamped, matching the
/// C int8 path. Inputs to quantized layers are tanh/sigmoid/GRU outputs in
/// `[-1, 1]`, so this never overflows in practice.
#[inline]
fn quant_x(x: f32) -> i8 {
    ((0.5 + 127.0 * x).floor() as i32).clamp(-127, 127) as i8
}

/// Walk the sparse 8×4 index structure, calling `f(group_base, pos, weight_off)`
/// once per column block. `weight_off` is the start of the block's 32 weights.
fn for_each_block(idx: &[i32], rows: usize, mut f: impl FnMut(usize, usize, usize)) {
    let mut wi = 0usize;
    let mut ii = 0usize;
    let mut i = 0usize;
    while i < rows {
        let cols = idx[ii] as usize;
        ii += 1;
        for _ in 0..cols {
            let pos = idx[ii] as usize;
            ii += 1;
            f(i, pos, wi);
            wi += 32;
        }
        i += SPARSE_ROWS;
    }
}

/// Dense float GEMV as a sequence of SAXPYs (j outer, i inner): streams the
/// matrix sequentially (cache-friendly, auto-vectorisable) while preserving the
/// per-`out[i]` accumulation order, so it is bit-identical to the C reference.
fn dense_sgemv(out: &mut [f32], w: &[f32], m: usize, n: usize, input: &[f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for (j, &xj) in input[..m].iter().enumerate() {
        let row = &w[j * n..j * n + n];
        for (o, &wv) in out.iter_mut().zip(row.iter()) {
            *o += wv * xj;
        }
    }
}

/// Largest input width of a quantized layer (conv2 and the GRUs are all 384).
const MAX_Q8_INPUTS: usize = 384;

/// int8 dense GEMV: quantize the activations, then `out[o] = scale[o] · (q_w[o] · q_x)`
/// with an int32 dot product (NEON `sdot` when available, scalar otherwise — both
/// give the exact same integer result, so output is independent of the path).
fn dense_q8(out: &mut [f32], w: &[i8], scale: &[f32], m: usize, n: usize, input: &[f32]) {
    let mut qx = [0i8; MAX_Q8_INPUTS];
    for (q, &x) in qx[..m].iter_mut().zip(&input[..m]) {
        *q = quant_x(x);
    }
    let qx = &qx[..m];

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: guarded by runtime detection of the `dotprod` feature.
            unsafe { dense_q8_neon(out, w, scale, m, n, qx) };
            return;
        }
    }
    dense_q8_scalar(out, w, scale, m, n, qx);
}

fn dense_q8_scalar(out: &mut [f32], w: &[i8], scale: &[f32], m: usize, n: usize, qx: &[i8]) {
    for o in 0..n {
        let row = &w[o * m..o * m + m];
        let mut acc = 0i32;
        for (&wv, &xv) in row.iter().zip(qx) {
            acc += wv as i32 * xv as i32;
        }
        out[o] = acc as f32 * scale[o];
    }
}

/// int8 dot product of two equal-length slices, accumulated in i32 (NEON `sdot`
/// when available, scalar otherwise — identical result). Used by the legacy
/// model's GRU input matmuls.
#[cfg(feature = "legacy-model")]
pub(crate) fn i8_dot(w: &[i8], x: &[i8]) -> i32 {
    let n = w.len();
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: guarded by runtime `dotprod` detection.
            return unsafe { i8_dot_neon(w, x, n) };
        }
    }
    let mut acc = 0i32;
    for k in 0..n {
        acc += w[k] as i32 * x[k] as i32;
    }
    acc
}

#[cfg(all(feature = "legacy-model", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
unsafe fn i8_dot_neon(w: &[i8], x: &[i8], n: usize) -> i32 {
    use core::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
    let mut k = 0;
    while k + 16 <= n {
        acc = sdot(
            acc,
            vld1q_s8(w.as_ptr().add(k)),
            vld1q_s8(x.as_ptr().add(k)),
        );
        k += 16;
    }
    let mut sum = vaddvq_s32(acc);
    while k < n {
        sum += *w.as_ptr().add(k) as i32 * *x.as_ptr().add(k) as i32;
        k += 1;
    }
    sum
}

/// Single `sdot` (signed int8 dot-product, 4 MACs/lane) via inline asm — the
/// `vdotq_s32` intrinsic is still unstable on stable Rust, but the instruction
/// is reachable directly. Requires the `dotprod` target feature.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn sdot(
    acc: core::arch::aarch64::int32x4_t,
    a: core::arch::aarch64::int8x16_t,
    b: core::arch::aarch64::int8x16_t,
) -> core::arch::aarch64::int32x4_t {
    let mut acc = acc;
    core::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dense_q8_neon(out: &mut [f32], w: &[i8], scale: &[f32], m: usize, n: usize, qx: &[i8]) {
    use core::arch::aarch64::*;
    let nchunks = m / 16;
    for o in 0..n {
        let row = w.as_ptr().add(o * m);
        let mut acc = vdupq_n_s32(0);
        let mut k = 0usize;
        for _ in 0..nchunks {
            let wv = vld1q_s8(row.add(k));
            let xv = vld1q_s8(qx.as_ptr().add(k));
            acc = sdot(acc, wv, xv);
            k += 16;
        }
        let mut sum = vaddvq_s32(acc);
        while k < m {
            sum += *row.add(k) as i32 * qx[k] as i32;
            k += 1;
        }
        out[o] = sum as f32 * scale[o];
    }
}

/// Port of `sparse_sgemv8x4` (float weights). `out` length is `rows`.
fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let mut wi = 0usize;
    let mut ii = 0usize;
    let mut i = 0usize;
    while i < rows {
        let cols = idx[ii] as usize;
        ii += 1;
        // Fixed-length subslice lets the compiler drop bounds checks and
        // vectorise the 8-wide row update cleanly.
        let out8 = &mut out[i..i + SPARSE_ROWS];
        for _ in 0..cols {
            let pos = idx[ii] as usize;
            ii += 1;
            let xs = &x[pos..pos + 4];
            let (xj0, xj1, xj2, xj3) = (xs[0], xs[1], xs[2], xs[3]);
            let wb = &w[wi..wi + 32];
            for r in 0..SPARSE_ROWS {
                out8[r] += wb[r] * xj0 + wb[8 + r] * xj1 + wb[16 + r] * xj2 + wb[24 + r] * xj3;
            }
            wi += 32;
        }
        i += SPARSE_ROWS;
    }
}

/// `compute_generic_dense`: linear + activation.
pub(crate) fn compute_dense(layer: &LinearLayer, out: &mut [f32], input: &[f32], act: Activation) {
    layer.compute(out, input);
    compute_activation(&mut out[..layer.nb_outputs], act);
}

/// `compute_generic_conv1d`: shift `mem` + `input` through the kernel window,
/// apply the linear layer, then activation, and update `mem`.
pub(crate) fn compute_conv1d(
    layer: &LinearLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    act: Activation,
) {
    let nb_inputs = layer.nb_inputs;
    let hist = nb_inputs - input_size;
    // Largest conv input in the model is conv2's 384.
    let mut tmp_buf = [0.0f32; MAX_CONV_INPUTS];
    let tmp = &mut tmp_buf[..nb_inputs];
    tmp[..hist].copy_from_slice(&mem[..hist]);
    tmp[hist..].copy_from_slice(&input[..input_size]);
    layer.compute(out, tmp);
    compute_activation(&mut out[..layer.nb_outputs], act);
    mem[..hist].copy_from_slice(&tmp[input_size..input_size + hist]);
}

/// `compute_generic_gru`: one GRU step, updating `state` in place.
pub(crate) fn compute_gru(
    input_w: &LinearLayer,
    recurrent_w: &LinearLayer,
    state: &mut [f32],
    input: &[f32],
) {
    let n = recurrent_w.nb_inputs;
    debug_assert_eq!(3 * n, recurrent_w.nb_outputs);
    debug_assert_eq!(input_w.nb_outputs, recurrent_w.nb_outputs);

    // All GRUs in the model have n = 384 (3*n = 1152).
    let mut zrh_buf = [0.0f32; MAX_GRU_3N];
    let mut recur_buf = [0.0f32; MAX_GRU_3N];
    let zrh = &mut zrh_buf[..3 * n];
    let recur = &mut recur_buf[..3 * n];
    input_w.compute(zrh, input);
    recurrent_w.compute(recur, state);

    for i in 0..2 * n {
        zrh[i] += recur[i];
    }
    compute_activation(&mut zrh[..2 * n], Activation::Sigmoid); // z = zrh[0..n], r = zrh[n..2n]

    for i in 0..n {
        let r = zrh[n + i];
        zrh[2 * n + i] += recur[2 * n + i] * r;
    }
    compute_activation(&mut zrh[2 * n..3 * n], Activation::Tanh); // h

    for i in 0..n {
        let z = zrh[i];
        let h = zrh[2 * n + i];
        state[i] = z * state[i] + (1.0 - z) * h;
    }
}

// Model layer dimensions (from rnnoise_data.h).
const CONV1_OUT: usize = 128;
const CONV1_IN: usize = 65;
const CONV1_STATE: usize = 65 * 2; // 130
const CONV2_OUT: usize = 384;
const CONV2_IN: usize = 128;
const CONV2_STATE: usize = 128 * 2; // 256
const GRU_OUT: usize = 384;
const CAT_SIZE: usize = CONV2_OUT + 3 * GRU_OUT; // 1536

/// Per-stream recurrent state (`RNNState`): conv kernel memories + GRU states.
pub(crate) struct RnnState {
    conv1_state: [f32; CONV1_STATE],
    conv2_state: [f32; CONV2_STATE],
    gru1_state: [f32; GRU_OUT],
    gru2_state: [f32; GRU_OUT],
    gru3_state: [f32; GRU_OUT],
}

impl RnnState {
    pub(crate) fn new() -> Self {
        RnnState {
            conv1_state: [0.0; CONV1_STATE],
            conv2_state: [0.0; CONV2_STATE],
            gru1_state: [0.0; GRU_OUT],
            gru2_state: [0.0; GRU_OUT],
            gru3_state: [0.0; GRU_OUT],
        }
    }
}

/// Port of `compute_rnn`: conv1→conv2→gru1→gru2→gru3, concatenate
/// `[conv2, gru1, gru2, gru3]`, then the gains (`dense_out`) and VAD heads.
pub(crate) fn compute_rnn(
    model: &crate::weights::RnnModel,
    rnn: &mut RnnState,
    gains: &mut [f32],
    vad: &mut [f32],
    input: &[f32],
) {
    let mut tmp = [0.0f32; CONV1_OUT];
    let mut cat = [0.0f32; CAT_SIZE];

    compute_conv1d(
        &model.conv1,
        &mut tmp,
        &mut rnn.conv1_state,
        input,
        CONV1_IN,
        Activation::Tanh,
    );
    compute_conv1d(
        &model.conv2,
        &mut cat,
        &mut rnn.conv2_state,
        &tmp,
        CONV2_IN,
        Activation::Tanh,
    );

    compute_gru(
        &model.gru1_input,
        &model.gru1_recurrent,
        &mut rnn.gru1_state,
        &cat[..CONV2_OUT],
    );
    compute_gru(
        &model.gru2_input,
        &model.gru2_recurrent,
        &mut rnn.gru2_state,
        &rnn.gru1_state,
    );
    compute_gru(
        &model.gru3_input,
        &model.gru3_recurrent,
        &mut rnn.gru3_state,
        &rnn.gru2_state,
    );

    cat[CONV2_OUT..CONV2_OUT + GRU_OUT].copy_from_slice(&rnn.gru1_state);
    cat[CONV2_OUT + GRU_OUT..CONV2_OUT + 2 * GRU_OUT].copy_from_slice(&rnn.gru2_state);
    cat[CONV2_OUT + 2 * GRU_OUT..].copy_from_slice(&rnn.gru3_state);

    compute_dense(&model.dense_out, gains, &cat, Activation::Sigmoid);
    compute_dense(&model.vad_dense, vad, &cat, Activation::Sigmoid);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activations_bounds() {
        assert!((tanh_approx(0.0)).abs() < 1e-6);
        assert!((sigmoid_approx(0.0) - 0.5).abs() < 1e-6);
        assert!(tanh_approx(100.0) <= 1.0 && tanh_approx(100.0) > 0.999);
        assert!(tanh_approx(-100.0) >= -1.0 && tanh_approx(-100.0) < -0.999);
    }

    // The `sdot` inline-asm path must produce exactly the same result as the
    // portable scalar dot product (both are exact int32 arithmetic).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sdot_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let (m, n) = (384usize, 70usize);
        let mut s = 7u32;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as i32 as i8
        };
        let w: Vec<i8> = (0..n * m).map(|_| next()).collect();
        let qx: Vec<i8> = (0..m).map(|_| next()).collect();
        let scale: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 1e-4).collect();
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        dense_q8_scalar(&mut a, &w, &scale, m, n, &qx);
        unsafe { dense_q8_neon(&mut b, &w, &scale, m, n, &qx) };
        assert_eq!(a, b, "sdot asm path diverged from scalar");
    }
}
