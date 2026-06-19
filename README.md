# rnnoise-rs

A from-scratch, dependency-free Rust port of Xiph
[**RNNoise**](https://gitlab.xiph.org/xiph/rnnoise) — real-time speech noise
suppression based on a recurrent neural network.

Unlike the well-known [`nnnoiseless`](https://github.com/jneem/nnnoiseless)
crate, which ports the *original* 2018 RNNoise (a 22-band, ~215 K-parameter
single-dense + 3-GRU model), `rnnoise-rs` ports the **current** (2024) RNNoise
architecture: a 32-band DSP front-end feeding a **2×Conv1D + 3×GRU**,
~2.9 M-parameter network. That model is a large step up in denoising quality.

## Highlights

- **Bit-exact parity** with the upstream C reference. On the test signal, the
  output differs from the C build by **3 samples in 47 520, each by 1 LSB**
  (relative energy `2e-10`) — well within the rounding spread between two C
  builds of RNNoise itself.
- **Faster than the C reference.** Same model, same machine (Apple M4 Pro),
  100 s of audio: `rnnoise-rs` 3.13 s vs C `-O3` (NEON) 4.59 s vs C scalar
  4.90 s — about **1.5× faster**, at **~32× real time** single-threaded. Full
  methodology and a vs-`nnnoiseless` comparison are in [BENCHMARKS.md](BENCHMARKS.md).
- **Self-contained:** the default model is embedded; no network or build step.
- **No dependencies** in the core library. Pure safe Rust (the only `unsafe` is
  in the optional C ABI).
- Optional **C ABI** (`capi` feature) compatible with `rnnoise.h`.

## Usage

```rust
use rnnoise::{DenoiseState, FRAME_SIZE};

let mut st = DenoiseState::new();
let mut out = [0.0f32; FRAME_SIZE]; // 480 samples
// `input` is 480 samples of 48 kHz mono audio, amplitudes ~ i16 range.
let input = [0.0f32; FRAME_SIZE];
let vad_probability = st.process_frame(&mut out, &input);
```

Audio is processed in **480-sample frames at 48 kHz** (10 ms). Sample
amplitudes follow the upstream convention: roughly the range of 16-bit PCM
(`i16` values cast to `f32`), **not** `[-1, 1]`. `process_frame` returns the
voice-activity probability and has a **one-frame algorithmic delay** (the output
corresponds to the previous input frame).

### Command-line demo

Mirrors upstream `rnnoise_demo` (RAW 16-bit machine-endian mono PCM @ 48 kHz):

```sh
cargo run --release --bin rnnoise-demo -- noisy.raw denoised.raw
```

### Custom models

Load any upstream RNNoise weight blob (as produced by `dump_weights_blob`):

```rust
use std::sync::Arc;
use rnnoise::{DenoiseState, RnnModel};

let model = Arc::new(RnnModel::from_bytes(&std::fs::read("weights_blob.bin")?)?);
let mut st = DenoiseState::with_model(model);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## "Better than nnnoiseless"?

It depends what you measure:

| | `nnnoiseless` (old model) | `rnnoise-rs` (current model) |
|---|---|---|
| Parameters | ~215 K | ~2.9 M |
| Denoising quality | good | **much better** (newer model) |
| Per-frame cost | ~32 µs | ~316 µs |
| Real-time factor | ~300× | ~32× |

The current model is ~13× larger, so it cannot match the tiny old model on raw
throughput — **no** faithful port of current RNNoise can. What `rnnoise-rs`
delivers is the much higher **quality** of the modern model, implemented faster
than the official C library, and still comfortably real-time (32× on one core).
See [BENCHMARKS.md](BENCHMARKS.md) for the full comparison and a speed-up study.

### Smaller footprint (int8)

`RnnModel::quantized()` returns an int8 version of the model: ~4× smaller weight
footprint (≈11.5 MB → ≈2.9 MB) with near-identical output (rel. energy `3.9e-6`
vs the float model). It is **not** bit-exact and, on wide-bandwidth CPUs like
Apple Silicon, not faster (this workload is compute-bound, not bandwidth-bound —
see BENCHMARKS.md). Use it to cut memory, e.g. with many concurrent streams:

```rust
use std::sync::Arc;
use rnnoise::{DenoiseState, RnnModel};
let model = Arc::new(RnnModel::default().quantized());
let mut st = DenoiseState::with_model(model);
```

## How it works

Per 10 ms frame: DC-removal high-pass → windowed 960-pt FFT → 32 ERB-style band
energies → pitch analysis (downsample + LPC whitening, coarse/fine correlation
search, doubling removal) → 65-dim feature vector (band-energy + pitch-correlation
cepstra + pitch lag) → Conv1D×2 + GRU×3 network predicting 32 band gains and a
VAD probability → pitch-comb filtering and gain smoothing on a one-frame-delayed
spectrum → inverse FFT and windowed overlap-add.

The FFT (a faithful port of the Opus KISS-FFT variant), the activation
approximations, the quantisation-free float matmuls and all the
double-precision quirks are reproduced exactly, which is what yields bit-level
parity. See [TODO.md](TODO.md) for the full architecture notes and parity
methodology.

## C API

Build a `cdylib`/`staticlib` with the C ABI:

```sh
cargo build --release --features capi
```

`include/rnnoise.h` declares the exported functions
(`rnnoise_create`, `rnnoise_process_frame`, `rnnoise_destroy`,
`rnnoise_model_from_buffer`, …), matching the upstream header for the
create/process/destroy flow.

## License

BSD-3-Clause, matching upstream RNNoise. The embedded model weights are the
official Xiph RNNoise model. RNNoise is © Jean-Marc Valin / Mozilla / Xiph.Org.
