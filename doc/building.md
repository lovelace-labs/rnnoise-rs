# Building, features & tooling

## Requirements

- Rust ≥ 1.74 (the crate's `rust-version`). No nightly needed — the NEON `sdot`
  kernel uses stable inline `asm!`, not the unstable intrinsic.
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
[parity.md](parity.md). One ignored timing test exists:
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
