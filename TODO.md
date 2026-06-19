# rnnoise-rs — Port Plan & TODO

A from-scratch, idiomatic Rust port of **current** [RNNoise](https://gitlab.xiph.org/xiph/rnnoise)
(the 2024 conv+GRU architecture in `bup/rnnoise-main/`), targeting:

1. **Full algorithmic parity** with the upstream C library (`rnnoise_process_frame`).
2. **Better denoising quality than `nnnoiseless`** — `nnnoiseless` ports the *old* (2018, 22-band,
   single-input-dense + 3-GRU, ~215 K param) model. We port the *new* (32-band, 2×Conv1D + 3×GRU,
   ~2.9 M param) model, which is a large quality improvement. We also aim to be the *fastest*
   correct implementation of the new model (SIMD, good memory layout).

> **Note on "perform better":** the new model is ~13× more matmul work per frame than the old one,
> so raw per-frame throughput cannot beat `nnnoiseless`'s tiny model — but the new model is far
> higher quality, and we will still run many× faster than real time. If the intent was instead a
> *faster* port of the *old* model, flag it and we pivot. Proceeding with the new model = "parity".

## Architecture (verified from upstream source + official model)

- Frame: **480 samples @ 48 kHz** (10 ms). Window 960 (50% overlap). 960-pt FFT → 481 bins.
- **32 ERB-ish bands** (`eband20ms`). Features = `2*32+1 = 65`:
  DCT(logBandEnergy)[32] + DCT(bandCorr w/ pitch)[32] + pitch_index[1].
- Pre: DC-removal biquad high-pass.
- Pitch: downsample(2×) → LPC(4) whitening FIR → coarse(4×)/fine(2×) xcorr search → remove_doubling.
- **NN** (loadable model, `init_rnnoise`): all values verified from `rnnoise_data.h/.c`:
  - conv1: dense **float**, in=195 (=65×ktime3), out=128, tanh. state 130.
  - conv2: in=384 (=128×3), out=384, tanh. Ships int8+float; C uses **float**. state 256.
  - gru1/2/3: input(384→1152) + recurrent(384→1152, +diag), sparse-8×4 **float** layout w/ idx.
  - cat = [conv2_out(384), gru1(384), gru2(384), gru3(384)] = 1536.
  - dense_out: 1536→32 (gains, sigmoid). vad_dense: 1536→1 (vad, sigmoid).
- Output: 1-frame **lookahead delay**; pitch_filter on delayed spectrum; per-band gains with
  RT60 decay cap (α=0.6) + cross-frame energy compensation; IFFT + windowed overlap-add.

## Parity strategy

- Oracle: upstream C compiled **scalar** (`-DDISABLE_NEON`, `RTCD_ARCH=c`) → `/tmp/rnnoise_ref/rnnoise_demo`.
  Matches the C `_c` path: `tanh_approx`/`sigmoid_approx` (vec.h rational polys), `sgemv`/`sparse_sgemv8x4`.
- Default model = official weights, **float path** (what the reference computes). Tables generated at
  init from the exact upstream formulas (window, dct) for bit-match; FFT = faithful kiss_fft port.
- Tests: per-stage intermediate dumps (FFT, bands, features, gains) vs instrumented C; end-to-end
  output vs `ref_out.raw` (target: `sum(diff²)/sum(ref²) < 1e-6`, i.e. tighter than nnnoiseless's 1e-4).

## Model / weights

- Official model downloaded + checksum-verified (`model_version` hash). Extracted to
  `/tmp/rnnoise_model_extract/`. Embeds via a generated native blob (`"DNNw"` records, 64-byte
  aligned) at `models/rnnoise_default.bin` (float-only slim blob ≈ 12 MB).
- Support loading custom models from the upstream blob format (`rnnoise_model_from_file` interop).
- Stretch: optional int8 (`weights_blob` ~3.5 MB) and the "little" sparse model behind features /
  for crates.io 10 MB-limit publishing.

---

## TODO

### Phase 0 — Setup & oracle
- [x] Study upstream C (denoise/rnn/nnet/pitch/celt_lpc/kiss_fft/vec) + nnnoiseless.
- [x] Download & checksum official model; learn exact dims from `rnnoise_data.h/.c`.
- [x] Build scalar C reference oracle; generate `ref_out.raw` from `testing.raw`.
- [ ] Generate slim float model blob → `models/rnnoise_default.bin`.
- [ ] Add per-stage dump hooks to the C oracle for intermediate parity checks.

### Phase 1 — Crate scaffold
- [ ] `Cargo.toml` (lib + bin + bench + features: `capi`, `little-model`), `.gitignore`.
- [ ] `src/lib.rs` module layout + public constants.

### Phase 2 — DSP front-end
- [ ] `fft.rs`: kiss_fft port (960-pt mixed-radix complex FFT), forward/inverse transforms.
- [ ] `common.rs`: window + dct tables (exact formulas), `dct`.
- [ ] bands: `compute_band_energy`, `compute_band_corr`, `interp_band_gain`.
- [ ] `biquad` HP filter; `apply_window`; `frame_analysis`; `frame_synthesis`.

### Phase 3 — Pitch
- [ ] `celt_lpc.rs`: `autocorr`, `lpc` (Levinson), `celt_fir5`, inner-prod helpers.
- [ ] `pitch.rs`: `pitch_downsample`, `pitch_xcorr`, `find_best_pitch`, `pitch_search`, `remove_doubling`.
- [ ] `pitch_filter`.

### Phase 4 — Features
- [ ] `compute_frame_features` (65-dim) + silence detection (E<0.04).

### Phase 5 — Neural net
- [ ] `nnet.rs`: `LinearLayer` (dense float `sgemv`, sparse 8×4 float, diag), `conv1d`, `gru`, `dense`.
- [ ] activations: `tanh_approx`, `sigmoid_approx` (match vec.h scalar exactly).
- [ ] `weights.rs`: native blob parser (`"DNNw"` records) + `RnnModel`/`init_rnnoise` wiring.
- [ ] embed default model; `from_file`/`from_bytes`/`from_static_bytes`.

### Phase 6 — Pipeline
- [ ] `DenoiseState` + `process_frame` (delay, RT60 cap, energy comp, pitch_filter, synthesis).

### Phase 7 — API & parity
- [ ] Rust API (`DenoiseState::new`, `process_frame`); docs.
- [ ] `capi.rs` matching `rnnoise.h` (feature `capi`, cbindgen header).
- [ ] Parity tests vs `ref_out.raw` + per-stage intermediates; lock tolerance in CI.

### Phase 8 — Performance
- [ ] SIMD hot paths (NEON + AVX2) for `sgemv`/`sparse_sgemv8x4`/conv; runtime feature detect.
- [ ] Criterion benches; compare throughput vs `nnnoiseless`; document quality vs speed.

### Phase 9 — Polish
- [ ] CLI demo bin (raw 48 kHz mono, like `rnnoise_demo`); optional wav via `hound`.
- [ ] README, rustdoc, examples, CI (fmt/clippy/test), LICENSE (BSD-3, match upstream).
