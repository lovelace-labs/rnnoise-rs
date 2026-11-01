# Building, features & tooling

## Requirements

- Rust ≥ 1.89 (the crate's `rust-version`), because the AVX-512 VNNI tier uses the
  AVX-512 intrinsics and the `avx512f` target feature, stable from 1.89. No nightly
  needed — the NEON `sdot` kernel uses stable inline `asm!` instead of the unstable
  intrinsic.
- No system dependencies for the default build. The `legacy-model` feature pulls
  in `realfft` (which brings `rustfft`); nothing else.

## Build

```sh
cargo build                      # library (current model), zero deps
cargo build --release            # optimized
cargo build --all-features       # + C ABI + legacy model
```

The release profile is tuned in `Cargo.toml` (`opt-level=3`, `lto=true`,
`codegen-units=1`); use `--release` for any real audio or benchmarking.

## Cargo features

| Feature | Default | Adds |
|---|:--:|---|
| *(none)* | ✓ | the current model; pure safe Rust, no deps |
| `capi` | | a C ABI (`#[no_mangle]` exports) matching `rnnoise.h` |
| `legacy-model` | | the original 2018 model (`DenoiseStateV1`); dep: `realfft` |

```toml
[dependencies]
rnnoise-rs = { version = "0.1", features = ["legacy-model"] }
```

## Runtime dispatch, not build-time flags

The SIMD int8 dot kernels are chosen per CPU at runtime (CPUID + cached feature
flags, once per matmul call), so a plain `cargo build` binary still uses AVX2 on an
AVX2 CPU and the AVX-512 VNNI kernel on a Zen 4 / Ice Lake+ CPU. `#[target_feature]`
is applied to the kernel functions only, never to the crate, so nothing requires
`-C target-cpu=x86-64-v3` or `-C target-cpu=native`. Those flags still let the compiler
contract `a*b + c` into an FMA on the float path, which changes the rounding per step — the
float tiers deliberately do not do that, and measured it is not a reason to deviate: on a Zen
3 core the explicit FMA kernels give ≤1.2× on the row-major shapes and nothing on the
block-major ones (see [performance.md](performance.md#fma-and-sse41-on-x86-64-measured-and-not-shipped)).

## Crate outputs

`[lib] crate-type = ["lib", "staticlib", "cdylib"]`, lib name `rnnoise`:

- **`lib`** — the normal Rust `rlib` for `use rnnoise::…`.
- **`staticlib`** — `librnnoise.a` for static linking into C/C++.
- **`cdylib`** — `librnnoise.{so,dylib,dll}` for dynamic linking.

The C symbols are only present with `--features capi`.

## CLI demo

```sh
cargo run --release --bin rnnoise-demo -- <noisy.raw> <denoised.raw>
```

RAW 16-bit machine-endian mono PCM @ 48 kHz in and out (not WAV), matching
upstream `rnnoise_demo` (it drops the first output frame). To make a test input
from a WAV, convert with e.g. `ffmpeg -i in.wav -f s16le -ar 48000 -ac 1 in.raw`.

## Tests

```sh
cargo test                  # default-feature tests (incl. C-reference parity)
cargo test --all-features   # + legacy-model + capi
```

What runs: end-to-end parity vs the C reference and int8 accuracy
(`tests/parity.rs`), legacy parity (`tests/legacy.rs`, needs `legacy-model`),
public-API behaviour (`tests/api.rs`), and unit tests in `fft`/`nnet`. See
[parity.md](parity.md).

The `simd`/`fft` tier tests skip a tier whose CPU does not have the feature, and print which
tiers they exercised (`simd/tests.rs`), so a run that cannot cover tier v4 says so instead of
looking green. One ignored timing test exists:
`cargo test --release --lib fft_timing -- --ignored --nocapture`.

## Benchmarks

```sh
cargo bench --bench process                      # current model: float + int8
cargo bench --features legacy-model --bench process
cargo run --release --features legacy-model --example bench_legacy   # vs nnnoiseless
```

The benches are dependency-free custom harnesses (`harness = false`). See
[performance.md](performance.md) and [`BENCHMARKS.md`](../BENCHMARKS.md).

## C ABI

```sh
cargo build --release --features capi
```

Link `target/release/librnnoise.a` (or the `cdylib`) and include
[`include/rnnoise.h`](../include/rnnoise.h). Exported symbols:
`rnnoise_get_frame_size`, `rnnoise_create`, `rnnoise_destroy`,
`rnnoise_process_frame`, `rnnoise_model_from_buffer`,
`rnnoise_model_from_filename`, `rnnoise_model_free`. Use the
create/process/destroy flow (see [api.md](api.md#c-api-capi-feature)). Minimal C:

```c
#include "rnnoise.h"
DenoiseState *st = rnnoise_create(NULL);     // default model
float in[480], out[480];
/* fill in[] with 48 kHz mono samples in i16 range … */
float vad = rnnoise_process_frame(st, out, in);
rnnoise_destroy(st);
```

## Lint & format

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features      # render the rustdoc
```

The codebase is clippy-clean at `-D warnings`. A few lints are intentionally
allowed crate-wide for bit-parity reasons (`manual_clamp`, `excessive_precision`)
— see the `#![allow(...)]` in [`src/lib.rs`](../src/lib.rs).

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on push/PR:
`fmt --check`, `clippy --all-targets --all-features -D warnings`, `build`, `test`,
and a release/parity test pass.

The tier-equality tests skip a tier whose CPU feature is not present, so a green run is not
coverage of that tier — GitHub-hosted runners do not reliably have AVX-512. The gate therefore
has two parts: the run prints the runner's ISA flags and the tiers it exercised (so coverage is
visible in the log), and a second job, `simd-tier-coverage`, runs on a real AVX-512 + VNNI
machine when one is wired up through the repository variable `avx512_runner` (a runner label).
It fails if that CPU does not report both features, and fails if a tier was skipped; without
the variable it is skipped rather than queued forever, because a hosted runner's CPU is not
something a workflow can choose. What can be proven without the hardware — tier v4's
`128 · Σ q_x` bias identity, and every tier's accumulation order against `simd::scalar` — is
asserted unconditionally, so it is gated on every runner.

## Repository layout

```
src/            library modules (see internals.md for the map)
  bin/          rnnoise-demo CLI
include/        rnnoise.h (C ABI header)
models/         embedded weights (default .bin, legacy .rnn)
test_data/      testing.raw + golden outputs (parity oracles)
benches/        process.rs throughput harness
examples/       bench_legacy.rs (fair legacy vs nnnoiseless)
scripts/        gen_blob.c (regenerate the default model blob)
doc/            this documentation
bup/            upstream RNNoise + nnnoiseless sources (reference; git-ignored)
```
