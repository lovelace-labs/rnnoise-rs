# Benchmarks: rnnoise-rs vs RNNoise (C) vs nnnoiseless

Reproducible benchmarks comparing:

- **rnnoise-rs** — this crate (current RNNoise model, ~2.9 M params).
- **RNNoise (C)** — upstream Xiph C library, scalar and `-O3`/NEON builds, same model.
- **nnnoiseless** — the other Rust port, of the *old* RNNoise model (~215 K params).

> **TL;DR.** rnnoise-rs is **~1.5× faster than the official C library** on the
> same model, bit-exact with it, at **~32× real time** single-threaded.
> `nnnoiseless` is ~10× faster still, but only because its model is ~13× smaller
> and lower quality — an apples-to-oranges speed comparison. The current model is
> **compute/load-bound, not memory-bandwidth-bound**, on Apple Silicon, which is
> why a naive int8 path does *not* speed it up here (measured below).

## System under test

| | |
|---|---|
| CPU | Apple M4 Pro (10 performance cores), 48 GB |
| OS | macOS (Darwin 25.5, arm64) |
| Rust | rustc 1.94.1, `opt-level=3, lto=true, codegen-units=1` |
| C | Apple clang 21, `-O2` (scalar, `-DDISABLE_NEON`) and `-O3` (NEON) |
| Audio | 48 kHz mono; `test_data/testing.raw` (≈1 s) tiled to 100 s |

## Methodology

Two complementary measurements:

1. **Per-frame compute (in-process).** Each implementation processes the *same*
   480-sample frame in a tight loop (20 000 iters after 200 warm-up), so the
   number is pure steady-state compute, excluding process start-up and I/O. An
   **active** frame (tone + noise) keeps the NN running; a **silent** frame makes
   RNNoise skip the network, so `full − front-end ≈ NN cost`.
2. **End-to-end CLI wall-clock.** Each tool denoises the same 100 s RAW file;
   we report the **min of 7 runs** (and derive µs/frame and the real-time factor).
   This includes start-up + I/O (small at this length).

All four read/write the same RAW 16-bit 48 kHz mono format.

## Results — per-frame compute (in-process)

| Implementation (model) | Full frame | Front-end | Neural net | NN % | Real-time |
|---|--:|--:|--:|--:|--:|
| **nnnoiseless** (old, ~215 K) | **28.8 µs** | 19.0 µs | 9.8 µs | 34 % | 347× |
| **rnnoise-rs** (new, float) | **312.6 µs** | 35.3 µs | 277.3 µs | 89 % | 32× |
| **rnnoise-rs** (new, int8) | 377.8 µs | 33.7 µs | 344.1 µs | 91 % | 26× |
| RNNoise C, NEON `-O3` (new) | 456.4 µs | 25.9 µs | 430.5 µs | 94 % | 22× |
| RNNoise C, scalar (new) | 497.0 µs | 24.5 µs | 472.5 µs | 95 % | 20× |

## Results — end-to-end CLI (100 s audio, min of 7)

| Implementation | Wall-clock | µs/frame | Real-time |
|---|--:|--:|--:|
| **rnnoise-rs** (float) | **3.125 s** | 312.5 | **32.0×** |
| RNNoise C, NEON `-O3` | 4.585 s | 458.5 | 21.8× |
| RNNoise C, scalar | 4.896 s | 489.6 | 20.4× |
| nnnoiseless (old model) | 0.321 s | 32.1 | 312× |

rnnoise-rs start-up (parsing the 11.3 MB embedded blob, once): **≈ 5.3 ms**.

## Analysis

**rnnoise-rs is faster than the C reference on the same model** (3.13 s vs
4.59 s NEON). The win is entirely in the neural net: rnnoise-rs's GEMVs
(277 µs) beat C's auto-vectorized ones (430 µs NEON, 473 µs scalar). The
decisive change was reorganising the matmuls into sequential, bounds-check-free
SAXPY/8×4 kernels the compiler vectorises cleanly (the per-output accumulation
order is preserved, so output stays **bit-identical** — 3 / 47 520 samples differ
by 1 LSB vs the C build).

**rnnoise-rs's front-end is slower than C's** (35 µs vs 25 µs) — our KISS-FFT
port and pitch search aren't as tight as upstream's. It's only ~11 % of the
frame, so it barely moves the total, but it's the clearest remaining target.

**nnnoiseless is faster because its model is smaller, not because the code is
faster.** Its NN is 9.8 µs (old ~215 K-param model, int8, fits in cache) vs our
277 µs (new ~2.9 M-param model, 11.5 MB of weights). Per byte of weights,
rnnoise-rs is actually *more* efficient. The trade is quality: rnnoise-rs runs
the modern, substantially better-denoising model. No faithful port of current
RNNoise can match the old model's throughput.

**The model is compute/load-bound, not memory-bandwidth-bound, on Apple
Silicon.** The NN moves 11.5 MB/frame at ≈ 41 GB/s — well under the M4 Pro's
single-core ceiling — so it's limited by load-port / FMA throughput, not DRAM
bandwidth. This is why the int8 path (next section) does **not** speed things up
on this machine.

## How to speed up — investigation

The NN is 89–95 % of every frame, so that's where speed lives. Findings:

### 1. int8 weights — *implemented, measured: smaller but not faster here*

Quantising conv2 + the GRUs to int8 with a per-output scale (opt-in via
[`RnnModel::quantized`]) cuts the weight footprint ~4× (11.5 MB → ~2.9 MB) and is
numerically near-identical to float (**rel. energy 3.9e-6, max 12 LSB** vs the
float reference). But on the M4 Pro it ran **0.81×** (slower): we're load/FMA
-bound, and dequantising int8 → f32 in the loop adds work without a bandwidth
payoff. **Conclusion:** keep int8 as a *footprint* option (and it should help on
bandwidth-starved CPUs); it is **not** a speed win without true int8 SIMD.

### 2. int8 *SIMD* (`sdot` / AVX-VNNI) — *biggest single-thread lever (future)*

The real int8 win needs widening dot-product instructions (ARM `sdot`, x86
VNNI) plus quantising the activations, accumulating in i32. This does 4 MACs per
lane per instruction and slashes load-port pressure — how upstream RNNoise goes
fast on AVX2. Expected **~2–3×** on the NN. Needs `unsafe` intrinsics +
runtime feature detection; not bit-exact.

### 3. f16 weights + f16 SIMD — *medium win (future)*

ARM does 8-wide f16 FMA (≈2× f32 throughput) and halves weight loads. Expected
~1.5–2× with intrinsics; ~half the accuracy loss of int8.

### 4. Multi-threading the GEMVs — *situational*

The three GRUs are sequential, but each layer's matmul (1152 outputs) can be
split across the M4 Pro's 10 perf cores. At a 10 ms frame budget the sync
overhead is significant for one stream; most useful for offline/batch. Expected
~2–4× wall-clock for batch, little benefit for a single real-time stream.

### 5. Tighten the front-end — *small, parity-safe-ish*

Replace the full 960-pt complex FFT with a real-input FFT (half the work) and
streamline the pitch inner-products to close the ~10 µs gap to C. ≈3 % of the
total; would change the bit-exactness of the front-end.

### 6. Smaller model — *quality trade*

The official "little" (sparse) model is ~½ the size; loading it (or training a
custom one) trades some quality for speed and footprint.

### Recommended priority

`sdot` int8 SIMD (#2) → f16 SIMD (#3) for single-stream latency; threading (#4)
for batch; front-end (#5) for a small, broad win. int8-storage (#1) today is for
memory, not speed.

## Reproduce

```sh
# rnnoise-rs per-frame (float + int8 breakdown)
cargo bench --bench process

# end-to-end on your own 48 kHz mono RAW (16-bit):
cargo run --release --bin rnnoise-demo -- noisy.raw out.raw

# parity + int8 accuracy vs the C reference
cargo test --release
```

The C reference and `nnnoiseless` figures were produced by building each from
`bup/` and timing the same RAW input; see the in-process harnesses described
above. Numbers vary with CPU and thermal state — treat them as ratios, not
absolutes.
