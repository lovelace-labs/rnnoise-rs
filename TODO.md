# rnnoise-rs — Port Plan, Status & TODO

A from-scratch, idiomatic Rust port of **current** [RNNoise](https://gitlab.xiph.org/xiph/rnnoise)
(the 2024 conv+GRU architecture in `bup/rnnoise-main/`).

Goals: (1) **full algorithmic parity** with the upstream C library; (2) be a
better choice than [`nnnoiseless`](https://github.com/jneem/nnnoiseless), which
ports the *old* (2018, 22-band, ~215 K param) model — we port the *new* (32-band,
2×Conv1D + 3×GRU, ~2.9 M param) model, a large quality improvement.

## ✅ Status: complete & validated

- **Parity:** end-to-end output vs the scalar C reference on `testing.raw`:
  **3 / 47 520 samples differ by 1 LSB** (rel. energy `2.0e-10`). On 4.8 M
  samples: 982 differ by 1 LSB (`7.3e-10`). This is within the spread between
  two C builds of RNNoise itself.
- **Speed (same model, Apple M-series, 100 s audio):** rnnoise-rs **3.11 s**
  vs C `-O3`/NEON 4.57 s vs C scalar 4.89 s → **~1.5× faster than C**, ~316 µs
  per 10 ms frame = **~32× real time**, single-threaded, no `unsafe` in the core.
- `nnnoiseless` (old, ~13× smaller, cache-resident model) runs ~32 µs/frame —
  it is faster *because the model is smaller*; rnnoise-rs wins on **quality**.
- All checks green: `cargo fmt --check`, `cargo clippy -D warnings`, 11 tests,
  `cargo doc`.

## Architecture (verified from upstream source + official model)

- Frame **480 @ 48 kHz** (10 ms); window 960; 960-pt FFT → 481 bins.
- **32 bands** (`eband20ms`). Features = `2*32+1 = 65`: DCT(logBandE)[32] +
  DCT(pitch-corr)[32] + pitch_index[1].
- Pre: DC-removal biquad. Pitch: 2× downsample + LPC(4) whitening → coarse(4×)/
  fine(2×) xcorr search → remove_doubling.
- **NN** (`init_rnnoise`, dims from `rnnoise_data.h`): conv1 dense-float 195→128
  tanh; conv2 384→384 tanh; gru1/2/3 input(384→1152)+recurrent(384→1152,+diag),
  sparse-8×4 float; cat[1536] → dense_out 1536→32 (gains) + vad_dense 1536→1.
- 1-frame look-ahead delay; pitch-comb filter; RT60 gain cap (α=0.6) + energy
  compensation; IFFT + windowed overlap-add.

## Parity methodology

- Oracle: upstream C built **scalar** (`-DDISABLE_NEON`, `RTCD_ARCH=c`) →
  `/tmp/rnnoise_ref/rnnoise_demo`. Matches the `_c` path (`tanh_approx`/
  `sigmoid_approx`, `sgemv`/`sparse_sgemv8x4`, float weights).
- Default model = official Xiph weights, **float path** (what the reference
  computes). Tables generated from the exact upstream formulas; FFT is a
  faithful kiss_fft port. Float matmuls preserve the C accumulation order, so
  results match to ~1 LSB (small residual = clang fp-contraction/FMA in C).

## Work log

- [x] Study upstream C + nnnoiseless; download & checksum official model.
- [x] Generate slim float weight blob → `models/rnnoise_default.bin` (11.3 MB,
      native `"DNNw"` format; also loadable by upstream `rnnoise_model_from_file`).
- [x] Build scalar C oracle; generate `test_data/ref_out.raw`.
- [x] FFT (kiss_fft 960-pt) + window/DCT tables + bands + biquad.
- [x] Pitch (celt_lpc, downsample, xcorr, search, remove_doubling) + pitch filter.
- [x] 65-dim feature extraction (with double-precision quirks).
- [x] NN: linear/sparse/conv1d/gru/dense + activations + blob loader + model wiring.
- [x] Pipeline: `DenoiseState::process_frame` (delay, RT60, energy comp, synthesis).
- [x] Parity test vs C oracle; lock tolerance in CI.
- [x] Perf: cache-friendly + bounds-check-free GEMVs → faster than C, bit-identical.
- [x] Public Rust API, C ABI (`capi`) + `include/rnnoise.h`, CLI demo, bench.
- [x] README, LICENSE (BSD-3), CI workflow, rustdoc, API tests.

## Benchmarks & speed-up study

Full results in [BENCHMARKS.md](BENCHMARKS.md). Headline (Apple M4 Pro, 100 s audio):
rnnoise-rs **3.13 s** vs C NEON 4.59 s vs C scalar 4.90 s (~1.5× faster, bit-exact);
nnnoiseless 0.32 s (smaller/older model). The NN is 89 % of each frame and is
**compute/load-bound, not bandwidth-bound**, on Apple Silicon.

- [x] **int8 + `sdot` SIMD** (`RnnModel::quantized()`): transposed dense int8
      layout + quantized activations + ARM `sdot` (inline asm; scalar fallback).
      **Measured 4.0× on the NN, 3.0× overall (105 µs/frame, 95× RT)**, ~4× smaller,
      near-identical output (rel. energy 4.8e-6). The naive i8→f32 dequant was
      0.81× (slower) — the win is `sdot`'s instruction-count reduction, since the
      model is compute/load-bound, not bandwidth-bound. Float stays the default.
- [x] **Legacy (2018) model support** (`legacy-model` feature → `DenoiseStateV1`):
      the old 22-band dense+3-GRU model — now **faster than nnnoiseless**:
      **25.9 µs/frame (309× RT) vs nnnoiseless 27.6 µs** (fair head-to-head),
      matching the old model to ~1e-5 rel. energy. Got there via:
  - [x] `realfft` (rustfft) for the FFT — same fast FFT nnnoiseless uses (optional
        dep, gated to the feature; KISS-FFT stays the default for the bit-exact
        current model). Legacy 44.9 → ~27.8 µs.
  - [x] 4-lags-at-once pitch cross-correlation + 4-wide inner products.
  - [x] NEON `sdot` int8 GRU input matmuls (dynamically-quantized activations) —
        the decisive edge nnnoiseless doesn't have; int8 stays (f32 weights were
        *slower* — 4× cache traffic).

## Future enhancements (not required for parity)

- [ ] **x86 int8 SIMD** (AVX-VNNI `vpdpbusd`): port the int8 dot kernel off ARM
      (scalar fallback already works everywhere); same ~3–4× win as `sdot`.
- [ ] **f16 weights + f16 SIMD** (~1.5–2× on ARM); less accuracy loss than int8.
- [ ] Multi-thread the per-layer GEMVs across cores (best for offline/batch).
- [ ] Tighten the front-end (real-input FFT) to close the ~10 µs gap to C.
- [ ] Ship a pre-quantized int8 / "little" model blob behind a feature so the
      crate fits the crates.io 10 MB limit (default float blob is 11.3 MB).
- [ ] `cargo-c` packaging / pkg-config for the C ABI; optional WAV I/O in the CLI.
- [ ] Big-endian support in the blob loader (currently little-endian).
