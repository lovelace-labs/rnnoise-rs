# rnnoise-rs documentation

`rnnoise-rs` is a from-scratch, idiomatic Rust port of Xiph
[RNNoise](https://gitlab.xiph.org/xiph/rnnoise) — real-time, single-channel
speech noise suppression driven by a recurrent neural network. It tracks the
**current** (2024) RNNoise architecture (a 32-band DSP front-end feeding a
2×Conv1D + 3×GRU network) and is numerically equivalent to the upstream C
library.

This folder is the long-form documentation. For the quick pitch and a usage
snippet, see the top-level [`README.md`](../README.md); for live API docs run
`cargo doc --open`.

## Contents

| Document | What's in it |
|---|---|
| [architecture.md](architecture.md) | The end-to-end algorithm, the per-frame data flow, and how the crate's modules map onto it. Start here. |
| [api.md](api.md) | How to use the library: the Rust API (`DenoiseState`, `RnnModel`), Cargo features, the C ABI, and the CLI. |
| [models.md](models.md) | The bundled models, the weight-blob format, loading custom models, the int8 quantized path, and the legacy model. |
| [internals.md](internals.md) | A module-by-module deep dive for contributors: FFT, bands, pitch, the neural net, weight loading and the pipeline. |
| [parity.md](parity.md) | How "numerically equivalent to upstream" is defined, measured and tested, and the exact results. |
| [performance.md](performance.md) | Throughput, the memory/compute profile, the SIMD/quantization optimizations, and how to reproduce the numbers. See also [`BENCHMARKS.md`](../BENCHMARKS.md). |
| [building.md](building.md) | Building, features, the CLI demo, the C/staticlib/cdylib artifacts, and CI. |

## At a glance

- **Frame model:** 480 samples of 48 kHz mono `f32` per call (10 ms), 50 %
  overlap, one frame of algorithmic latency. Amplitudes are in `i16` range
  (`-32768.0..=32767.0`), not `[-1, 1]`.
- **Quality:** the current ~2.9 M-parameter model — a large step up from the
  ~215 K-parameter 2018 model that `nnnoiseless` ports.
- **Parity:** bit-exact-class agreement with the C reference (3 of 47 520 test
  samples differ by 1 LSB).
- **Speed (Apple M4 Pro, single core):** ~32× real time (float, default),
  ~95× real time (opt-in int8 + `sdot`). The optional legacy model runs ~309×
  real time — faster than `nnnoiseless`.
- **Dependencies:** none for the default build. The only `unsafe` is the
  optional C ABI and the optional NEON `sdot` kernel.

## Provenance & license

The algorithm and the embedded model weights are the work of Jean-Marc Valin /
Mozilla / Amazon / Xiph.Org (and Mark Borgerding for KISS-FFT). This is a Rust
port distributed under the same BSD-3-Clause terms; see [`LICENSE`](../LICENSE).
