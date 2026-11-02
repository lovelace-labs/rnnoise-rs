# Parity with upstream

"Numerically equivalent to the C library" is a core goal of this port. This
document defines what that means, how it's measured, and the exact results.

## What "parity" means here

True bit-exactness across implementations isn't even well-defined for RNNoise:
the C library's own output depends on the build (scalar vs SSE vs AVX2 vs NEON),
because the SIMD paths use different reductions and different `tanh`/`sigmoid`
approximations. Two C builds of RNNoise already differ from each other by ~1 LSB
on a few samples.

So the target is: **match the scalar C reference (`_c` path) as built by the pinned
recipe**, which the port does bit for bit. The port replicates the scalar path
operation-for-operation (the `vec.h` `tanh_approx`/`sigmoid_approx`, the
`sgemv`/sparse GEMV accumulation order, and all the float↔double promotions), which
is what gets it there. A 1-LSB allowance is kept only where a *different compiler or
libc* is compared with committed bytes — two C builds already differ by ~1 LSB on a
few samples — never between the port and the goldens.

## The oracle

The reference is upstream RNNoise compiled **scalar**:

- Sources from `bup/rnnoise-main/` plus the official model
  (`rnnoise_data.c`, checksum-verified against `model_version`).
- Compiled with `RTCD_ARCH = c` **and the pure-C branch of `vec.h`**: `-U__SSE2__
  -U__AVX__` on x86-64 (the compiler predefines `__SSE2__`, so `vec.h` would
  otherwise include `vec_avx.h` — a different `tanh_approx`/`sigmoid_approx` and a
  different `sgemv`, i.e. different roundings) and `-DDISABLE_NEON` on AArch64.
  Without both the build is not the path this crate ports: against the goldens it
  disagrees by ~1800 1-LSB samples instead of a handful.
- Run on `test_data/testing.raw` (≈1 s of 48 kHz speech) to produce
  `test_data/ref_out.raw` — the committed golden output.

`scripts/c_reference.sh` does all of that from a **pinned commit**
(`70f1d256acd4b34a572f999a05c87bf00b67730d`) and the recorded model hash, and CI
runs it on x86-64, aarch64 and Apple Silicon. `tests/c_reference_parity.rs` then
compares the port and the freshly built oracle against the committed goldens for
every clip. It skips gracefully when the oracle or the clips are absent, so an
offline `make check` stays green. It is gated to `embedded-model`: the goldens are
the float model's output, so a build whose default model is the int8 blob cannot be
checked against them (see [models.md](models.md#int8-quantized-model) and
`tests/parity.rs::quantized_accuracy` for the int8 tolerance).

The float-only model blob this crate embeds was cross-checked against the C
library's *embedded* model by loading it through the C `rnnoise_model_from_file`
path: the two outputs differed by only 6 of 47 520 samples (1 LSB each), i.e. the
slim float blob and the full embedded model are equivalent.

## Results (current model)

`tests/parity.rs::matches_c_reference` processes `testing.raw`, drops the first
frame (the look-ahead), converts to `i16` like the C demo (truncation toward
zero), and compares to `ref_out.raw`:

```
samples = 47520
samples differing      = 0
max difference         = 0
sum(diff²)/sum(ref²)   = 0
```

The test asserts `rel < 1e-6` — far tighter than `nnnoiseless`'s `1e-4` acceptance — and,
because the golden is the pinned recipe's output, **equality**. A differing sample is a
behaviour change that has to be justified against the reference, not a rounding to absorb.

### The residual that used to be here, and how it was closed

It was not FP contraction: a pure-C build at `-O0 -ffp-contract=off` differed by the same
magnitude, and clang at `-O2` with contraction enabled produced the same output as gcc with
it disabled. Instrumenting the feature vector on `testing.raw` (C vs port, frame by frame)
showed **exactly one feature ever differing** — feature 64, `features[2*NB_BANDS]`:

- C: `features[2*NB_BANDS] = .01*(pitch_index-300)` — `.01` is a `double`, so the product
  is computed in double and narrowed to float.
- Port: `0.01f32 * (pitch_index as f32 - 300.0)` — computed in `f32`.

Those last-bit differences propagated through the network and landed on 3 output samples.
Everything else — twiddles, `log10`, the DCT, band energy/corr, the pitch filter — was bit
identical between the pinned oracle and the port, frame by frame. Computing that one op in
double closed it: the port is bit-exact against the pinned oracle on every clip now, and the
gate asserts that. The single op is worth the rule it enforces (C1: mirror the reference
operation for operation), and `src/legacy.rs` applies the same promotion for the same reason
— there the measured numbers do not move, because the legacy path quantizes its GRU input to
int8 anyway.

The clip set is `test_data/testing.raw` (speech) plus three deterministic clips
from `scripts/make_clips.py`; every golden is now the pinned oracle build's output:

| clip | what it exercises | port vs golden | fresh C build vs golden |
|---|---|---|---|
| `testing.raw` | real speech | 0 — bit-exact | 0 |
| `clips/tone_noise.raw` | stable pitch + noise floor | 0 — bit-exact | 0 |
| `clips/silence.raw` | near-silence front-end | 0 — bit-exact | 0 |
| `clips/zero.raw` | silence, network skipped | 0 — bit-exact | 0 |

The gate is equality between the port and the goldens, and a 1-LSB spread between the
goldens and a fresh build on another compiler. The golden the earlier clang build produced
is kept as `test_data/goldens/testing.c-clang.raw` for provenance; against it the port shows
4 samples, 1 LSB, `2.7e-10` — the difference is that build's FP contraction, not the port.

The Rust float tiers never contract either, so the numbers above hold for every tier
`simd::dispatch` can select. A build with `-C target-feature=+fma` produced the same bytes as
the non-contracting build on the machine used here, but that is luck per operation, not a
property: a fused step is one rounding where the reference does two, and on a
cancellation-heavy synthetic sum it can accumulate to ~1e-5 absolute. That is why the ladder
does not contain an FMA tier. See
[performance.md](performance.md#fma-and-sse41-on-x86-64-measured-and-not-shipped).

The CLI demo reproduces this end-to-end: `./rnnoise-demo testing.raw out.raw` is
byte-identical to the C demo's output on the same clip.

## int8 quantized path

`tests/parity.rs::quantized_accuracy` runs `RnnModel::default().quantized()` and
compares to the same float reference:

```
rel. energy = 4.8e-6   (max 13 LSB)
```

Not bit-exact (it quantizes weights and activations to int8), but audibly
identical — that's why it's opt-in and float is the default. The `sdot` NEON
kernel and the scalar fallback are proven to produce **identical** integer results
by `simd::tests::sdot_matches_scalar`, so the int8 output doesn't depend on
whether `dotprod` is available.

## Legacy (2018) model

The legacy pipeline targets the old model, whose true reference is the original C.
`nnnoiseless` (also a port of that model) matches the original C to `rel ≈ 1.7e-6`.
The oracle for this crate's legacy path is `nnnoiseless`'s own output on
`testing.raw` (`test_data/legacy_ref.raw`):

```
tests/legacy.rs::legacy_matches_old_model
rel. energy = 1.1e-5   (max 21 LSB)   — asserted < 1e-4
```

The residual is larger than the current model's because the legacy fast path
quantizes the GRU input activations to int8 for the `sdot` speed-up (the edge that
makes it faster than `nnnoiseless`). It is still imperceptible. With a plain
(non-quantized) FFT pipeline the legacy output matched the old model to ~1e-7, so
the ~1e-5 is entirely the quantization trade chosen for speed.

## Reproducing the oracle

The C reference build isn't committed (it needs the multi-MB upstream sources and
the downloaded model), but the recipe is:

1. Fetch the official model
   (`rnnoise_data-<hash>.tar.gz` from `media.xiph.org`, hash in `model_version`),
   extract `rnnoise_data.{c,h}`.
2. Compile `denoise.c rnn.c pitch.c celt_lpc.c kiss_fft.c nnet.c nnet_default.c
   parse_lpcnet_weights.c rnnoise_tables.c rnnoise_data.c examples/rnnoise_demo.c`
   with `clang -O2 -U__SSE2__ -U__AVX__ -DDISABLE_NEON -Iinclude -Isrc` — the
   `-U__SSE2__` is what selects `vec.h`'s pure-C branch on x86-64, see "The oracle".
   (The autotools default is the int8/OSCE compute path, i.e.
   `--disable-dnn-debug-float`; that is not this oracle — the float `_c` path is.)
3. Run it on `test_data/testing.raw`; that's `ref_out.raw`.

`make reference` (i.e. `scripts/c_reference.sh`) is that recipe, pinned to a commit
and a model hash, writing per-clip outputs to `.c_reference/`. Goldens can be
refreshed with `WRITE_GOLDENS=1 ./scripts/c_reference.sh`, and every committed golden
was produced exactly this way — which is what makes the equality assertion in
`tests/parity.rs` a fact about the port rather than a tolerance. The
`legacy_ref.raw` oracle was produced by running `nnnoiseless`'s library
`process_frame` on the same input.

## Per-tier coverage in CI

Parity is claimed for the scalar path, and the tiers are claimed to be bit-identical
to it. Two things check that on CI hardware:

- `src/simd/tests.rs` — each kernel against the portable reference, including the
  non-multiple-of-vector-width tails and the pure-Rust VNNI bias identity (which is
  the only way to validate v4 on a runner without AVX-512). These are what caught the
  VNNI row-length bug, and an end-to-end run on the real model could not:
  every real layer is full width, so the fixture widths are not replaceable.
- `src/simd/coverage.rs` — the whole pipeline over `testing.raw`, once per tier the
  runner reports, float and int8 paths, asserted bit-identical to the scalar run.

`RNNOISE_SIMD_TIER` forces a tier (`src/simd/tier.rs`). It is `cfg(test)`-only, or opt-in
through the `simd-tier-override` feature for `cargo bench`, so a normal build pays nothing
for it. It only takes effect when the CPU has the ISA — forcing AVX2 on a machine without
it would be an illegal instruction — and a tier that CI *requested* and cannot run is a
failure, not a skip, because the promise was coverage. Tiers that were not requested are
reported as skipped, so a CI run states explicitly which paths executed:

```text
simd coverage (100 frames, 47520 samples):
    scalar  float: bit-identical to scalar
    scalar  int8:  bit-identical to scalar
      sse2  float: bit-identical to scalar
      sse2  int8:  bit-identical to scalar
    sse4.1  float: bit-identical to scalar
    sse4.1  int8:  bit-identical to scalar
      avx2  float: bit-identical to scalar
      avx2  int8:  bit-identical to scalar
      vnni  skipped: runner has no vnni
```

## Why this matters

Because the port is operation-faithful, behaviour you rely on in upstream RNNoise
— the exact suppression curve, the VAD probability, the latency, the silence
handling — carries over. It also means a weight blob exported from upstream
(`dump_weights_blob`) drops straight into `RnnModel::from_bytes`, and a blob this
crate produces loads back into the C library.
