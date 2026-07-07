# Performance

This is the optimization-oriented guide: where the time goes, what was done about
it, and how to reproduce the numbers. For the full benchmark tables and
methodology see [`BENCHMARKS.md`](../BENCHMARKS.md).

All figures are Apple M4 Pro, single core, `opt-level=3, lto=true,
codegen-units=1`. Per-frame is the steady-state cost of one `process_frame`;
"real time" is how many times faster than the 10 ms a frame represents.

## Summary

| Configuration | per frame | real-time | notes |
|---|--:|--:|---|
| current model, float (default) | ~312 µs | ~32× | bit-exact with C, zero deps |
| current model, int8 + `sdot` | ~105 µs | ~95× | opt-in, audibly identical |
| legacy model (`legacy-model`) | ~26 µs | ~309× | **faster than `nnnoiseless`** |
| — `nnnoiseless` (same old model) | ~28 µs | ~290× | for reference |

The current model is ~13× larger than the old one, so it cannot match the tiny
old model on raw throughput — but it denoises far better, and it runs faster than
the official C library (~1.5× on the same model; see [BENCHMARKS.md](../BENCHMARKS.md)).

## Where the time goes

For the **current model**, the neural net is ~89 % of a frame; the DSP front-end
(FFT, bands, pitch, features) is the rest. The net moves ~11.5 MB of float weights
per frame, but at ~41 GB/s it's well under the M4 Pro's single-core ceiling — so
it's **instruction/load-bound, not DRAM-bandwidth-bound**. That distinction drives
every optimization decision below.

For the **legacy model**, the net is tiny (~10 µs) and the front-end (FFT + pitch)
dominates, so the legacy work targeted the FFT and pitch.

## What was done

### Cache-friendly, auto-vectorizing GEMV (current model, parity-safe)

The dense matmuls are written as sequential SAXPYs (`j` outer, `i` inner), which
stream the weight matrix in order and let the compiler vectorize, while preserving
the exact per-output accumulation order — so the result stays **bit-identical** to
the C reference. Reorganising the GEMVs this way (and giving the sparse 8×4 kernel
bounds-check-free fixed-length slices) made the float path **faster than the C
library** (≈1.5×), at full parity.

### int8 + NEON `sdot` (current model, opt-in)

`RnnModel::quantized()` stores conv2 and the GRUs as transposed dense int8 with a
per-output scale, quantizes activations to int8, and computes each output with a
NEON `sdot` (4 int8 MACs per lane per instruction). Measured: **NN 277 → 69 µs
(4.0×)**, **full frame 312 → 105 µs (3.0×)**, footprint ~4× smaller, accuracy
`rel 4.8e-6` (see [models.md](models.md#int8-quantized-model)).

Key lesson — because the workload is instruction-bound, *how* you do int8 matters:

| approach | result |
|---|---|
| naive int8 (dequantize to f32 in the loop) | **0.81×** — *slower* (adds instructions) |
| f32 weights (no conversion, 4× memory) | *slower* (4× cache traffic) |
| **int8 + `sdot`** (fewer instructions, int math) | **3–4× faster** |

`sdot` is emitted via stable inline `asm!` (the `vdotq_s32` intrinsic is still
nightly-only), guarded by runtime `is_aarch64_feature_detected!("dotprod")`, with
a scalar fallback that yields the identical integer result. Non-aarch64 targets
use the scalar path automatically.

### Beating `nnnoiseless` on the legacy model

The legacy model started ~45 µs (with the KISS-FFT) and is now ~26 µs — faster
than `nnnoiseless`'s ~28 µs — via three changes:

1. **`realfft` (rustfft) for the FFT.** The from-scratch KISS-FFT is ~2× slower
   per transform than rustfft; matching nnnoiseless's FFT closed most of the gap.
   `realfft` is an optional dependency, gated to the `legacy-model` feature, so the
   default crate stays dependency-free and the current model keeps its bit-exact
   KISS-FFT.
2. **Tighter pitch:** a 4-lags-at-once `pitch_xcorr` and 4-wide inner products.
3. **`sdot` int8 GRU input matmuls** with dynamically-quantized activations — the
   decisive edge nnnoiseless doesn't have (it makes the NN faster while keeping the
   low-memory int8 weights).

Fair head-to-head (standalone, isolated processes, median of many runs):
**rnnoise-rs ~26 µs vs nnnoiseless ~28 µs (~7 % faster).**

## Choosing a configuration

- **Default (float):** highest quality, bit-exact, zero deps, ~32× real time —
  fine for real-time on one core with lots of headroom.
- **int8 (`.quantized()`):** ~3× faster and ~4× smaller, imperceptible quality
  loss — use it for many concurrent streams, embedded, or battery-sensitive cases.
- **legacy (`legacy-model`):** smallest/fastest, lower quality — use only when you
  specifically want the old model's profile.

## Reproducing

```sh
# current model: float + int8 breakdown (and legacy if the feature is on)
cargo bench --bench process
cargo bench --features legacy-model --bench process

# fair, isolated legacy vs nnnoiseless comparison
cargo run --release --features legacy-model --example bench_legacy

# end-to-end on your own 48 kHz mono RAW
cargo run --release --bin rnnoise-demo -- noisy.raw out.raw
```

Numbers vary with CPU and thermal state — treat them as ratios, not absolutes.
The benches build a deterministic noisy frame and time `process_frame` in a tight
loop after warm-up; the `process` bench also reports a front-end-only (silent
input) figure so you can separate the NN from the DSP.

## Remaining levers

See [`BENCHMARKS.md`](../BENCHMARKS.md) §"How to speed up" and
[`TODO.md`](../TODO.md): x86 VNNI for the int8 kernel, f16 SIMD, multi-threading
the GEMVs for batch, and a real-input/SIMD FFT for the current model (which would
relax its bit-exactness bound).
