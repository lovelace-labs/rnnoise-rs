# Architecture

This document describes what `rnnoise-rs` computes for every frame and how the
source modules implement it. It is a faithful port of the current RNNoise
`rnnoise_process_frame` (the 2024 conv+GRU model), so the algorithm below is the
upstream algorithm; the value of the port is doing it exactly, in safe Rust,
fast.

## Constants

All defined in [`src/lib.rs`](../src/lib.rs):

| Name | Value | Meaning |
|---|--:|---|
| `FRAME_SIZE` | 480 | samples in/out per call (10 ms @ 48 kHz) |
| `WINDOW_SIZE` | 960 | analysis window (50 % overlap) |
| `FREQ_SIZE` | 481 | FFT bins kept (`WINDOW_SIZE/2 + 1`) |
| `NB_BANDS` | 32 | ERB-style perceptual bands |
| `NB_FEATURES` | 65 | NN input features (`2*NB_BANDS + 1`) |
| `PITCH_MIN_PERIOD` | 60 | shortest pitch period searched |
| `PITCH_MAX_PERIOD` | 768 | longest pitch period searched |
| `PITCH_FRAME_SIZE` | 960 | samples used for pitch correlation |
| `PITCH_BUF_SIZE` | 1728 | pitch history buffer (`MAX_PERIOD + FRAME_SIZE`) |

Bands are defined by `EBAND20MS` — bin edges
`[0, 2, 4, 6, 8, 10, 12, 15, …, 356, 400]` (50 Hz per bin; band 31 ends at
20 kHz). Bins 400..481 (20–24 kHz) are not modelled and end up silenced.

## Per-frame data flow

```
 input[480] (i16-range f32)
     │
     ▼  rnn_biquad (DC-removal high-pass, f64 state)         common::biquad
 x[480]
     │
     ├─────────────── frame analysis ─────────────────┐
     ▼                                                 │
 [analysis_mem | x]  → window(960) → FFT(960)          │   denoise::frame_analysis
     │                    → X[481]  → band energy Ex[32]│   common::{apply_window,
     │                                                  │     forward_transform},
     │                                                  │     compute_band_energy
     ▼  pitch analysis (on a 1728-sample history)       │
 downsample×2 → LPC(4) whiten → coarse(4×)/fine(2×)     │   pitch::{pitch_downsample,
   xcorr search → remove_doubling → pitch period T      │     pitch_search, remove_doubling}
     │                                                  │
     ▼  build pitch-lagged window p, transform           │
 P[481], Ep[32], Exp[32] (band correlation X·P)         │   denoise::compute_frame_features
     │                                                  │
     ▼  65-D feature vector                              │
 features = DCT(log Ex)[0..32]                           │   common::dct
          + DCT(Exp normalised)[32..64]                  │
          + 0.01*(T - 300)             [64]              │
     │   (if total energy E < 0.04 → "silence", skip NN)─┘
     ▼
 compute_rnn(features) ───────────────────────────────────  nnet::compute_rnn
   conv1(195→128, tanh)  ── kernel over 3 frames
   conv2(384→384, tanh)  ── kernel over 3 frames        ┐
   gru1(384→384) gru2 gru3  (each: input + recurrent)   │ all on a
   cat = [conv2(384), gru1, gru2, gru3] = 1536          │ 1-frame
   dense_out(1536→32, sigmoid) → g[32]   (band gains)   │ lookahead
   vad_dense(1536→1, sigmoid)  → vad probability        ┘
     │
     ▼  apply to the *previous* frame's spectrum (the delay)
 pitch_filter(delayed_X, delayed_P, …, g)               denoise::pitch_filter
 g[i] = max(g[i], 0.6*lastg[i])         (RT60 decay cap)
 lastg[i] = min(1, g[i]*(dEx+1e-3)/(Ex+1e-3))  (energy compensation)
 delayed_X *= interp_band_gain(g)                        common::interp_band_gain
     │
     ▼  synthesis
 IFFT(delayed_X) → window → overlap-add(synthesis_mem)   denoise::frame_synthesis
     │
     ▼
 output[480],  returns vad probability
```

### The one-frame look-ahead

`process_frame` returns the denoised version of the **previous** input frame.
Internally the gains computed from the current frame's features are applied to a
stored copy of the previous frame's spectrum (`delayed_X`, `delayed_P`,
`delayed_Ex/Ep/Exp`). This lets the suppression gains "see" one frame into the
future, which is why the very first output frame is all-zero and is conventionally
discarded (the CLI demo and tests drop it).

### Silence handling

If the summed band energy `E < 0.04`, the frame is treated as silence: the
feature vector is zeroed and the neural net (and pitch filter / gain smoothing)
are skipped, leaving the recurrent state untouched. The delayed spectra are still
rotated, so output continues seamlessly.

## Module map

| Module | Responsibility | Upstream origin |
|---|---|---|
| [`fft`](../src/fft.rs) | `KissFft` — mixed-radix complex FFT (960-pt); `RealFft` for the legacy model | `kiss_fft.c` |
| [`common`](../src/common.rs) | window + DCT tables, band energy/correlation, gain interpolation, biquad, forward/inverse transforms | `denoise.c`, `rnnoise_tables.c` |
| [`celt_lpc`](../src/celt_lpc.rs) | LPC (Levinson), autocorrelation, FIR-5, inner-product / cross-correlation kernels | `celt_lpc.c`, `pitch.h` |
| [`pitch`](../src/pitch.rs) | downsampling, coarse/fine pitch search, doubling removal | `pitch.c` |
| [`nnet`](../src/nnet.rs) | `LinearLayer` (dense/sparse/int8), Conv1D, GRU, activations, `compute_rnn`, `RnnState` | `nnet.c`, `nnet_arch.h`, `vec.h`, `rnn.c` |
| [`weights`](../src/weights.rs) | `RnnModel`, the `"DNNw"` blob parser, `ModelError`, the embedded default | `parse_lpcnet_weights.c`, `rnnoise_data.c` |
| [`denoise`](../src/denoise.rs) | `DenoiseState`, `process_frame`, feature extraction, pitch filter, synthesis | `denoise.c` |
| [`capi`](../src/capi.rs) | C ABI matching `rnnoise.h` (feature `capi`) | `rnnoise.h` |
| [`legacy`](../src/legacy.rs) | the original 2018 model as `DenoiseStateV1` (feature `legacy-model`) | old RNNoise / `nnnoiseless` |

Shared, immutable tables (`Common`: window, DCT matrix, FFT plan) live behind a
`OnceLock` and are built once for the whole process. The default model is parsed
once and shared via `Arc<RnnModel>`. Everything mutable and per-stream lives in
`DenoiseState`.

## The neural network in detail

Dimensions are taken from the official model (`rnnoise_data.h`) and verified by
the loader at startup:

```
features[65]
   │
   ▼  Conv1D  (kernel 3 frames → in 65×3 = 195, out 128, tanh)     conv1  (dense f32)
 tmp[128]
   │
   ▼  Conv1D  (kernel 3 frames → in 128×3 = 384, out 384, tanh)    conv2  (f32; int8 avail.)
 cat[0..384]
   │
   ├─ gru1: input 384→1152, recurrent 384→1152 (+diag), state 384  (sparse 8×4, +bias)
   ├─ gru2: same shape, fed by gru1 state
   └─ gru3: same shape, fed by gru2 state
   │
   ▼  cat = [conv2_out(384), gru1(384), gru2(384), gru3(384)] = 1536
   ├─ dense_out: 1536→32, sigmoid → g[32]  (per-band suppression gains)
   └─ vad_dense: 1536→1, sigmoid → voice-activity probability
```

A GRU step (`nnet::compute_gru`) is the standard update/reset/candidate form:

```
z = σ(Wz·in + Uz·state + bz)
r = σ(Wr·in + Ur·state + br)
h = tanh(Wh·in + Uh·(r ⊙ state) + bh)
state' = z ⊙ state + (1 - z) ⊙ h
```

where the `U*` recurrent matrices carry an extra learned diagonal term. Conv1D is
implemented as a linear layer over a sliding window of the last 3 frames
(`mem` holds the previous 2). Activations use the exact `vec.h` rational-
polynomial `tanh`/`sigmoid` approximations so results match the C `_c` path.

For how the weights are stored, quantized and loaded, see [models.md](models.md);
for the line-level implementation of each block, see [internals.md](internals.md).

## Relationship to the old (2018) model

The optional `legacy-model` feature provides the *original* RNNoise — a different
pipeline: 22 bands, a 42-D feature vector with cepstral Δ/ΔΔ history and spectral
variability, and an input-dense + 3-GRU network (no conv layers, table-based
activations). It shares this crate's pitch and biquad code but has its own bands,
features and network. See [models.md](models.md#legacy-2018-model) and
[performance.md](performance.md).
