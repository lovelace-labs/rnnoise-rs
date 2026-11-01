# Performance

This is the optimization-oriented guide: where the time goes, what was done about
it, and how to reproduce the numbers. For the full benchmark tables and
methodology see [`BENCHMARKS.md`](../BENCHMARKS.md).

All figures are Apple M4 Pro, single core, `opt-level=3, lto=true,
codegen-units=1`, except the x86-64 rows, which are an AMD Ryzen 9 5900X (Zen 3)
with plain `cargo bench` — i.e. the default `x86-64` (SSE2) codegen baseline plus
runtime feature detection, no `-C target-cpu`. Per-frame is the steady-state cost of one `process_frame`;
"real time" is how many times faster than the 10 ms a frame represents.

The Zen 3 figures are a head-to-head against the pre-SIMD baseline commit
[`74b46b1`](https://github.com/lovelace-labs/rnnoise-rs/commit/74b46b1065630f68464ee60dd542e3ce175743db):
same harness file, same deterministic bench frame, same toolchain (rustc 1.99.1),
one core pinned, the two builds run alternating in the same session, means of four
rounds. Run-to-run spread is ~5 %, so read these as ratios.

## Summary

| Configuration | per frame | real-time | notes |
|---|--:|--:|---|
| current model, float (default) | ~312 µs | ~32× | bit-exact with C, zero deps |
| current model, int8 + `sdot` (aarch64) | ~105 µs | ~95× | opt-in, audibly identical |
| current model, int8 + `pmaddwd` (x86-64, Zen 3) | ~100 µs | ~100× | runtime-detected AVX2; **4.9× vs the baseline** (which had no x86 int8 kernel at all) |
| current model, float, x86-64 Zen 3 | ~157 µs | ~63× | **1.36×** vs `4a26f93`, **1.64×** vs the baseline; see the x86-64 sections below |
| legacy model (`legacy-model`), M4 Pro | ~26 µs | ~309× | **faster than `nnnoiseless`** |
| legacy model, x86-64 Zen 3 | ~33 µs | ~299× | **1.21×** vs the baseline |
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
a scalar fallback that yields the identical integer result.

### x86-64 float tiers (parity-safe, runtime-detected)

The int8 tiers only help the quantized model, so the float path used to run at the
plain `x86-64` (SSE2, 4-lane) codegen rate even on an AVX2 machine. `src/simd` now has
float tiers for the three hot DSP/DNN loops, each of them **bit-identical to the portable
reference** because it keeps the reference's accumulation order and never fuses a multiply
with an add (no `vfmadd` appears in the disassembly). *Bit-identical to the reference as it
stands today*: the sparse 8×4 commit also re-grouped the reference itself, which moves a
couple of 1-LSB roundings relative to the baseline — see the head-to-head section below.

| kernel | what the tier does | measured on a Zen 3 core |
|---|---|--|
| `simd/x86_64/dense.rs` | 8 outputs per accumulator instead of 4 | 384×1152 layer 33.2 → 23.5 µs/frame |
| `simd/x86_64/xcorr.rs` | 4 (SSE2) / 8 (AVX2) lags in parallel, one accumulator per lag | coarse pitch search 7.2 → 3.0 µs/frame |
| `fft/x86_64.rs` | vector butterflies over the inner `j`, reading per-stage contiguous twiddle views built by the plan | 4.86 → 2.5 µs per transform (3 per frame) |

Two structural changes make those tiers pay off:

- **Complete sparse indices are dense-equivalent.** The shipped model's GRU matrices are
  stored in the sparse 8×4 layout, but every row group lists all `nb_inputs/4` blocks in
  ascending order. The walk then visits each output's contributions in ascending input
  order — exactly `dense_sgemv`'s order — so `weights.rs` loads such layers in dense
  row-major form at parse time (same bytes, same bits) and the GEMV can run through the
  wider tier. Out-of-order or genuinely sparse blobs keep the sparse kernel.
- **Band weights are precomputed** in `Common` (`j / band_size`, the same f32 division as
  the runtime did), removing ~317 float divisions per band pass.

End to end, plain `cargo bench`, tier v3 selected at runtime (best of runs):

| | before | after |
|---|--:|--:|
| float full frame | 270.6 µs | **208–216 µs** (46–48× real time) |
| — float front-end | 31.6 µs | **21.8 µs** |
| int8 full frame | 116.8 µs | **102–107 µs** (94–98× real time) |
| legacy full frame | 38.1 µs | **33.8 µs** |

Cross-check, same machine, four alternating rounds against the baseline commit (means,
not best of runs): float `305.2 → 205.2 µs`, front-end `31.4 → 21.7 µs` — see
[Head-to-head](#head-to-head-against-the-pre-simd-baseline-74b46b1).

Parity is unchanged by all of this (`3 / 47 520 samples, 1 LSB`, `2.003e-10`), which is
the point of keeping the accumulation order: the tiers are a layout and width change, not
an arithmetic change. Aarch64 keeps the portable reference for these loops, so its numbers
are unchanged.

### x86-64 int8 tiers (parity-safe, runtime-detected)

Before this, x86-64 had no int8 dot kernel at all: on the `x86-64` baseline codegen
(`SSE2`) a byte dot product cannot be vectorized — `pmaddubsw` is SSSE3 and
`vpdpbusd` is VNNI — so the int8 path was **slower** than float (487 vs 289 µs of
NN per frame).

`src/simd/x86_64/` now has one kernel per ISA tier, chosen at **runtime** by CPUID
(`src/simd/x86_64/select.rs`), once per matmul call, never through `-C target-cpu`:

| tier | ISA | kernel | MACs/cycle on this Zen 3 core |
|---|---|---|--:|
| v1 | SSE2 (x86-64 baseline) | `punpcklbw` + `cmpgt` sign mask + `pmaddwd` | 3.5 |
| v2 | SSE4.1 (x86-64-v2) | `pmovsxbw` + `pmaddwd` | 6.5 |
| v3 | AVX2 (x86-64-v3) | same shape, 16 lanes | **15.3** |
| v4 | AVX-512 VNNI (x86-64-v4) | `vpdpbusd`, unsigned × signed → biased + corrected | ~30 (no VNNI CPU here) |

The same idea applies to the sparse 8×4 GEMV: `x86_64/sparse8x4.rs` keeps the
eight-row accumulator in a ymm register for the whole row group instead of
reloading and storing it per block, and accumulates per column in the order
upstream `vec.h` uses, so it is bit-identical to `scalar::sparse_sgemv8x4`.

End to end, plain `cargo bench`, tier v3 selected at runtime:

| | before | after |
|---|--:|--:|
| int8 neural net | 487 µs | **84 µs** (5.8×) |
| int8 full frame | 519 µs | **119 µs** (84× real time) |
| legacy full frame | 40.2 µs | 38.1 µs |

Cross-check, same machine, four alternating rounds against the baseline commit (means):
int8 `509.5 → 104.3 µs` full frame, `477.9 → 82.6 µs` neural net, legacy `40.4 → 33.4 µs`.
At the baseline int8 was *slower* than float (0.58×), so the ladder is the whole
story for x86.

Tier v4 is available on Rust ≥ 1.89 (AVX-512 intrinsics + `avx512f` target feature),
which is why the crate's `rust-version` is 1.89. It does not force AVX-512 codegen:
a binary built with 1.89+ runs tier v3 on an AVX2-only CPU and v4 on a Zen 4.

Every int8 tier produces the **same exact integer** (products and pair sums fit in `i32`
for the model's widths), so the quantized output is unchanged — the accuracy test
still reports `rel_energy 4.826e-6`, and `src/simd/tests.rs` asserts tier == scalar
for every tier the CPU reports, skipping the others. The VNNI bias identity
(`Σ(w+128)·x − 128·Σx == Σ w·x`) is also unit-tested in pure Rust, so it is checked
even on runners without AVX-512.

### Block-major dense float GEMV (parity-safe, x86-64 tier)

In the row-major GEMV each output keeps its own accumulator, so a group of eight outputs
costs a load and a store of that accumulator on top of the multiply and add: 2 loads + 1
store + 2 FP ops per 8 MACs. Storing the same values **grouped per output block** — eight
outputs' weights contiguous over `j` — lets the tier keep the accumulator in a register for
the whole `j` walk, and running four groups at once puts four independent add chains in
flight. It is a permutation of the same floats, and every output still sums in ascending
`j`, so the bits are identical: `rnnoise-demo` on `test_data/testing.raw` outputs the same
bytes as before the change, and the parity harness reports the same numbers.

Measured on a Zen 3 core (same toolchain, one core pinned, isolated layer call):

| layer shape | row-major | block-major | ratio |
|---|--:|--:|--:|
| 384×1152 (the GRU matrices) | 20.7 µs | 15.3 µs | 1.35× |
| 384×384 (conv2) | 12.5 | 6.6 | 1.9× |
| 195×128 (conv1) | 2.2 | 0.8 | 2.7× |
| 1536×32 (dense_out) | 5.0 | 1.4 | 3.6× |

End to end, five alternating rounds (the comparison build is the committed tree, tier v3
selected at runtime):

| | before | after |
|---|--:|--:|
| float full frame | 213.5 µs | **157.4 µs** (46.8× → 63.5× real time) |
| — float neural net | 192.0 µs | **135.7 µs** (1.41×) |
| — float front-end | 21.5 µs | 21.7 µs (unchanged) |
| int8 full frame | 102.8 µs | 99.5 µs (layout does not apply: its dot is already contiguous per output) |
| end-to-end CLI, 100 s, min of 5 | 2.131 s | **1.593 s** (1.34×) |

Without a fused multiply-add the ceiling is 8 MACs/cycle (one 32-byte weight load plus one
mul and one add per group of eight). The row-major shape reaches ~5.1, the block shape ~6.7,
so the float GEMV is now within about 20% of what this core can do; the remaining step is
FMA, which changes the rounding and is therefore not allowed by the parity bound.

## Head-to-head against the pre-SIMD baseline (`74b46b1`)

Means of four alternating rounds on the pinned Zen 3 core, both builds from the same
toolchain and profile. These are the numbers at `4a26f93`; the block-major change above sits
on top of them.

| path | metric | `74b46b1` | HEAD (`4a26f93`) | speedup | real-time |
|---|--:|--:|--:|--:|--:|
| float | full frame | 305.2 µs | **205.2 µs** | **1.49×** | 32.8× → 48.7× |
| | front-end | 31.4 | 21.7 | 1.45× | 318× → 461× |
| | neural net | 273.7 | 183.4 | 1.49× | |
| int8 | full frame | 509.5 µs | **104.3 µs** | **4.88×** | 19.6× → 95.9× |
| | front-end | 31.6 | 21.7 | 1.46× | |
| | neural net | 477.9 | 82.6 | **5.79×** | 20.9× → 121× |
| legacy | full frame | 40.4 µs | **33.4 µs** | 1.21× | 248× → 299× |
| | neural net | 21.7 | 16.3 | 1.33× | |
| | front-end | 18.7 | 17.0 | 1.10× | |

At the baseline the int8 path was *slower* than float (0.58×) because x86 had no
byte dot kernel; on the branch int8 is 2.0× faster than float. End to end
(`rnnoise-demo`, 100 s of `test_data/testing.raw` tiled, min of 5 runs each):
**3.042 s → 2.001 s (1.52×)**, i.e. 301 → 198 µs/frame; the ~5 ms start-up is unchanged.

Per commit (same protocol, two rounds each — approximate, ±5 %):

| commit | float full | float front | float NN | int8 full | what moved |
|---|--:|--:|--:|--:|---|
| `74b46b1` | 305 | 31.4 | 274 | 509 | baseline |
| `b966e6d` | 305 | 31.5 | 274 | 115 | int8 tier ladder |
| `7014a44` | 264 | 30.2 | 234 | 115 | float dense + xcorr tiers |
| `389047e` | 221 | 30.0 | 191 | 113 | dense-form loading of complete sparse indices |
| `d9d1cbc` | 199 | 21.8 | 177 | 101 | FFT butterfly tier |
| `a3e09ea` | 201 | 21.6 | 180 | 103 | precomputed band weights (~1 µs) |
| `4a26f93` | 202 | 21.7 | 181 | 103 | docs |

The two builds are not byte-identical to each other, and it is worth stating why: 2 of
47 520 samples differ by 1 LSB (363 over the 100 s run, because the state carries
forward). The first divergence in a numeric bisect is the sparse 8×4 commit
(`ae79ade`, re-landed as `b966e6d`) — it re-grouped the accumulation from a four-term
chain (`w[r]·x0 + w[8+r]·x1 + …` summed in one expression) to one product per column
accumulated across the row group, which is the order upstream `vec.h` uses. So the tier is
bit-identical to the current portable reference (`src/simd/tests.rs` asserts that for every
tier the CPU reports), while the reference's own grouping changed against the baseline.
The parity harness reports the same numbers on both builds
(`samples=47520 nonzero_diff=3 max_diff=1 rel_energy=2.003e-10`), so nothing regressed.

## Choosing a configuration

- **Default (float):** highest quality, bit-exact with the C reference up to a handful of
  1-LSB roundings, zero deps, ~32× real time on M4 Pro (~63× on the Zen 3 core) —
  fine for real-time on one core with lots of headroom.
- **int8 (`.quantized()`):** ~3× faster than float on aarch64, ~4.9× faster than the
  pre-SIMD x86 baseline (~2.0× faster than float on this Zen 3 core), ~4× smaller,
  imperceptible quality loss — use it for many concurrent streams, embedded, or
  battery-sensitive cases.
- **legacy (`legacy-model`):** smallest/fastest, lower quality — use only when you
  specifically want the old model's profile.

## Reproducing

```sh
# current model: float + int8 breakdown (and legacy if the feature is on)
cargo bench --bench process
cargo bench --features legacy-model --bench process

# head-to-head against the pre-SIMD baseline: build the baseline in a separate
# checkout and run both alternately, pinned to one core
mkdir -p /tmp/baseline && git archive 74b46b1 | tar -x -C /tmp/baseline
cargo bench --bench process                       # HEAD
cd /tmp/baseline && cargo bench --bench process   # baseline
# run them alternately on the same core to keep thermal conditions equal:
#   taskset -c 2 cargo bench --bench process

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
the GEMVs for batch, and a real-input FFT for the current model (the SIMD tier is done;
a real-input transform would relax its bit-exactness bound).

Two measured but not yet implemented levers:

- **int8 activations pre-widened to int16.** The AVX2 dot widens `x` inside every row
  (2 µops per block). Widening once per call instead of once per row gives 7.06 → 6.47 µs
  per 384×1152 call (+8.4%) on this core, and integer arithmetic makes the two forms
  identical. It was left out because the `sdot` and VNNI tiers need the byte form, so the
  kernel would have to carry both.
- **8-lane butterflies for the FFT.** The tier is 4 lanes (SSE2) and the radix-2 stage is
  still scalar; an AVX2 version saves roughly 1.5 µs per transform, ~4 µs per frame.
