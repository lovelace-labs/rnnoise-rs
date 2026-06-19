# Benchmarks: rnnoise-rs vs RNNoise (C) vs nnnoiseless

Reproducible benchmarks comparing:

- **rnnoise-rs** — this crate (current RNNoise model, ~2.9 M params).
- **RNNoise (C)** — upstream Xiph C library, scalar and `-O3`/NEON builds, same model.
- **nnnoiseless** — the other Rust port, of the *old* RNNoise model (~215 K params).

> **TL;DR.** rnnoise-rs is **~1.5× faster than the official C library** on the
> same model, bit-exact with it, at **~32× real time** single-threaded — or
> **~95× real time** with the opt-in int8 + `sdot` path (3× faster, audibly
> identical). `nnnoiseless` is faster still, but only because its model is ~13×
> smaller and lower quality — an apples-to-oranges speed comparison. The model is
> **compute/load-bound, not memory-bandwidth-bound** on Apple Silicon, so a naive
> int8 path does *not* help, but `sdot` (4 int8 MACs/instruction) does.

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
| **rnnoise-rs** (old, `legacy-model`) | 38.4 µs | 29.0 µs | **9.3 µs** | 24 % | 261× |
| **rnnoise-rs** (new, float) | 312.6 µs | 35.3 µs | 277.3 µs | 89 % | 32× |
| **rnnoise-rs** (new, int8 + `sdot`) | **105.1 µs** | 34.0 µs | **68.9 µs** | 66 % | **95×** |
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
Silicon.** The float NN moves 11.5 MB/frame at ≈ 41 GB/s — well under the M4
Pro's single-core ceiling — so it's limited by load-port / FMA *instruction*
throughput, not DRAM bandwidth. That distinction is the whole story for int8: a
naive int8 path (dequantise to f32 in the loop) does **not** help (it adds
instructions), but `sdot` — which does 4 int8 MACs per lane in one instruction —
cuts instruction count ~4× and gives a real **4.0× speedup on the NN** (next
section).

## Legacy model vs nnnoiseless (apples-to-apples)

With the `legacy-model` feature, `rnnoise-rs` runs the *same* old model as
`nnnoiseless`, so this is a like-for-like comparison:

| | full | front-end | neural net |
|---|--:|--:|--:|
| nnnoiseless | 28.8 µs | 19.0 µs | 9.8 µs |
| rnnoise-rs `DenoiseStateV1` | 38.4 µs | 29.0 µs | **9.3 µs** |

The **neural net is a dead heat** (9.3 vs 9.8 µs) — `rnnoise-rs` is, if anything,
marginally faster there. The whole 9.6 µs gap is in the **front-end**: our pitch
search/bands and especially the FFT are less tuned than nnnoiseless's, which uses
a real-input FFT (`easyfft`). We reuse the complex KISS-FFT.

One front-end win is already applied: the old model needs the forward FFT of both
the signal and the pitch-lagged signal, so we compute them as a **single** complex
FFT (`z = signal + i·lagged`) and split the conjugate-symmetric spectrum back out
— two real FFTs for the price of one. That took the legacy frame from 44.9 → 38.4
µs (≈15 %) with **no** loss of parity (still 1.1e-7 vs the old model). Closing the
rest needs a real-input FFT for the remaining forward + the inverse (see §5).

Output parity (legacy): **rel. energy 1.1e-7, max 1 LSB** vs the old model —
*tighter* than nnnoiseless's own match to the original C reference (1.7e-6).

## How to speed up — investigation

The NN is 89–95 % of every frame, so that's where speed lives. Findings:

### 1. int8 + `sdot` SIMD — ✅ *implemented, 3× faster*

Quantising conv2 + the GRUs to int8 with a per-output scale (opt-in via
[`RnnModel::quantized`]) stores a transposed dense `[output][input]` matrix so
each output is a contiguous int8 dot product. Activations (all in `[-1, 1]` after
tanh/sigmoid/GRU) are quantised to int8, and each output is computed with ARM
`sdot` — 4 int8 MACs per lane per instruction — via stable inline `asm!` (the
`vdotq_s32` intrinsic is still nightly-only), with a scalar fallback that gives
the identical integer result.

Measured on M4 Pro: **NN 277 → 68.9 µs (4.0×)**, **full frame 312 → 105 µs
(3.0×, 95× real time)**, footprint ~4× smaller. Accuracy vs the float model:
**rel. energy 4.8e-6, max 13 LSB** — audibly identical. Not bit-exact, so it's
opt-in; the float model stays the default. (A naive dequant-to-f32 int8 loop, by
contrast, measured *0.81×* — slower — confirming the win is the instruction-count
reduction, not bandwidth.)

### 2. x86 VNNI / `vpdpbusd` — *port the int8 path off ARM (future)*

The same int8 design maps to AVX-512-VNNI / AVX-VNNI (`vpdpbusd`) on x86 for an
equivalent win; only the per-output dot kernel needs an x86 path (the scalar
fallback already works everywhere).

### 3. f16 weights + f16 SIMD — *medium win (future)*

ARM does 8-wide f16 FMA (≈2× f32 throughput) and halves weight loads. ~1.5–2×
with intrinsics, with ~half the accuracy loss of int8 — a middle ground when int8
error is undesirable.

### 4. Multi-threading the GEMVs — *situational*

The three GRUs are sequential, but each layer's matmul (1152 outputs) can be
split across the M4 Pro's 10 perf cores. At a 10 ms frame budget the sync
overhead is significant for one stream; most useful for offline/batch. Expected
~2–4× wall-clock for batch, little benefit for a single real-time stream.

### 5. Tighten the front-end — *partly done*

The front-end is now ~⅓ of an int8 frame (and most of a legacy frame). Done: the
two forward FFTs are computed as one complex FFT (legacy 44.9 → 38.4 µs, no parity
loss). Remaining: a real-input FFT for the last forward + the inverse (half the
work again) and tighter pitch inner-products — this would close the ~10 µs
front-end gap to nnnoiseless/C. A real-input FFT changes the front-end rounding,
so it would relax the new model's bit-exactness bound (the legacy model already
tolerates it).

### 6. Smaller model — *quality trade*

The official "little" (sparse) model is ~½ the size; loading it (or training a
custom one) trades some quality for speed and footprint.

### Recommended priority

int8 + `sdot` (#1) is done and is the big lever on ARM. Next: VNNI for x86 (#2),
then the real-input FFT (#5) for a broad front-end win; f16 (#3) and threading
(#4) as needed.

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
