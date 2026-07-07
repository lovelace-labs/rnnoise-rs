# Parity with upstream

"Numerically equivalent to the C library" is a core goal of this port. This
document defines what that means, how it's measured, and the exact results.

## What "parity" means here

True bit-exactness across implementations isn't even well-defined for RNNoise:
the C library's own output depends on the build (scalar vs SSE vs AVX2 vs NEON),
because the SIMD paths use different reductions and different `tanh`/`sigmoid`
approximations. Two C builds of RNNoise already differ from each other by ~1 LSB
on a few samples.

So the target is: **match the scalar C reference (`_c` path) to within a tiny,
fixed tolerance**, ideally to a handful of 1-LSB differences. The port replicates
the scalar path operation-for-operation (the `vec.h` `tanh_approx`/`sigmoid_approx`,
the `sgemv`/sparse GEMV accumulation order, and all the float↔double promotions),
which is what gets it there.

## The oracle

The reference is upstream RNNoise compiled **scalar**:

- Sources from `bup/rnnoise-main/` plus the official model
  (`rnnoise_data.c`, checksum-verified against `model_version`).
- Compiled with `-DDISABLE_NEON` and `RTCD_ARCH = c`, so it takes the `_c`
  compute path (the same one this crate ports), not NEON/AVX.
- Run on `test_data/testing.raw` (≈1 s of 48 kHz speech) to produce
  `test_data/ref_out.raw` — the committed golden output.

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
samples differing      = 3
max difference         = 1 LSB
sum(diff²)/sum(ref²)   = 2.0e-10
```

The test asserts `rel < 1e-6` and `max_diff <= 2` — far tighter than
`nnnoiseless`'s `1e-4` acceptance, and tighter than the spread between two C
builds. The residual 3 LSB come from clang contracting some `a*b + c` into FMAs in
the C build (Rust does not auto-contract), which is sub-ULP per op.

The CLI demo reproduces this end-to-end:
`./rnnoise-demo testing.raw out.raw` vs the C demo differ by the same 3 LSB.

## int8 quantized path

`tests/parity.rs::quantized_accuracy` runs `RnnModel::default().quantized()` and
compares to the same float reference:

```
rel. energy = 4.8e-6   (max 13 LSB)
```

Not bit-exact (it quantizes weights and activations to int8), but audibly
identical — that's why it's opt-in and float is the default. The `sdot` NEON
kernel and the scalar fallback are proven to produce **identical** integer results
by `nnet::tests::sdot_matches_scalar`, so the int8 output doesn't depend on
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
   with `clang -DDISABLE_NEON -O2 -I. -Iinclude`.
3. Run it on `test_data/testing.raw`; that's `ref_out.raw`.

The committed `test_data/ref_out.raw` was produced exactly this way. The
`legacy_ref.raw` oracle was produced by running `nnnoiseless`'s library
`process_frame` on the same input.

## Why this matters

Because the port is operation-faithful, behaviour you rely on in upstream RNNoise
— the exact suppression curve, the VAD probability, the latency, the silence
handling — carries over. It also means a weight blob exported from upstream
(`dump_weights_blob`) drops straight into `RnnModel::from_bytes`, and a blob this
crate produces loads back into the C library.
