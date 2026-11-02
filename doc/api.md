# API guide

This is the usage-oriented reference. For generated per-item docs run
`cargo doc --open`; for the algorithm see [architecture.md](architecture.md).

## Audio format

Everything is **48 kHz, mono, `f32`**, processed in fixed **480-sample frames**
(10 ms). Sample amplitudes follow the upstream convention: roughly the range of
16-bit PCM, i.e. `i16` values cast to `f32` (`-32768.0..=32767.0`) — **not**
`[-1.0, 1.0]`. If your pipeline uses `[-1, 1]` floats, multiply by `32768.0`
before and divide after.

`process_frame` has **one frame of algorithmic latency**: the output of a call
corresponds to the *previous* input frame, so the first output frame is all-zero
and is conventionally discarded.

## The Rust API

The crate root re-exports everything you normally need:

```rust
use rnnoise::{DenoiseState, RnnModel, ModelError, FRAME_SIZE};
```

Other public constants: `WINDOW_SIZE`, `FREQ_SIZE`, `NB_BANDS`, `NB_FEATURES`.

### `DenoiseState`

The streaming denoiser. One per audio stream (it holds the recurrent state and
overlap-add buffers); it is **not** `Sync`-shared across concurrent frames of the
same stream.

```rust
let mut st = DenoiseState::new();              // built-in model
// or: DenoiseState::new_with_model(model)      // a shared Arc<RnnModel>

let vad: f32 = st.process_frame(&mut out, &inp); // out,inp: &[f32] ≥ FRAME_SIZE
```

- `new()` / `Default` — use the embedded default model (shared across instances;
  parsed once).
- `new_with_model(model: Arc<RnnModel>)` — back the denoiser with a specific model.
  (`with_model()` is the old name, kept as a deprecated alias: `with_foo()` is reserved
  for closure accessors.)
- `process_frame(&mut self, output, input) -> f32` — denoise one frame; returns
  the voice-activity probability in `[0, 1]`. `output` and `input` must each be at
  least `FRAME_SIZE` long (longer slices: only the first `FRAME_SIZE` are used).
- `rnnoise::warmup()` — resolve the runtime CPU feature flags now instead of on the first
  call. `new_with_model()` already does it, so a caller that builds its state outside the
  audio callback does not need to call this; it exists for callers that construct the state
  inside the callback, or that want the one-time probe attributable to startup. It changes
  no output. See [realtime.md](realtime.md).

### Streaming whole signals

Process in `FRAME_SIZE` chunks and drop the first output frame:

```rust
use rnnoise::{DenoiseState, FRAME_SIZE};

fn denoise(samples: &[f32]) -> Vec<f32> {
    let mut st = DenoiseState::new();
    let mut out = [0.0f32; FRAME_SIZE];
    let mut acc = Vec::with_capacity(samples.len());
    let mut first = true;
    for chunk in samples.chunks_exact(FRAME_SIZE) {
        st.process_frame(&mut out, chunk);
        if !first {
            acc.extend_from_slice(&out);
        }
        first = false;
    }
    acc // trailing < FRAME_SIZE samples are dropped, as upstream does
}
```

### `RnnModel`

The immutable weights, shared across denoisers.

```rust
let model = RnnModel::default();                       // the embedded model (features)
let model = RnnModel::from_bytes(&blob)?;              // a custom weight blob
let model = RnnModel::from_file(Path::new("m.bin"))?;  // `model-loader` feature
let fast  = RnnModel::default().quantized();           // int8 variant (see below)
let bytes = model.to_bytes();                          // the mirror of from_bytes
```

- `default()` — parse the embedded model. Only exists with `embedded-model` or
  `embedded-model-int8`; the float blob wins when both are enabled, because feature
  unification can enable both without anyone asking. Panics only if the embedded blob is
  invalid, which is a build-time problem, not a runtime one.
- `from_bytes(&[u8]) -> Result<RnnModel, ModelError>` — load an upstream RNNoise weight
  blob (the `"DNNw"` format produced by `dump_weights_blob`), bare or inside the `RNNS`
  container, whose format version, record count and checksum are verified. See
  [models.md](models.md).
- `from_file(path)` and `from_file_with_checksum(path, expected_crc)`
  (`model-loader` feature) — the on-demand path for an application that ships no embedded
  blob. The pinned-checksum form is what catches a structurally plausible but wrong file;
  a CRC32 is a corruption check, not an authenticity check. Both are startup paths: they
  allocate and perform I/O, so they are not real-time-safe and must not run on the audio
  thread. An existing memory mapping already works with no extra API: `from_bytes(&mmap)`.
- `crc32(&[u8]) -> u32` — the CRC32 (IEEE, reflected) the container uses, exposed so a
  caller can pin a downloaded blob before parsing it.
- `to_bytes() -> Vec<u8>` — serialise the model as the `RNNS` container plus its records;
  round-trips through `from_bytes`.
- `quantized(self) -> RnnModel` — return an int8-quantized copy: ~4× smaller and
  ~3× faster, audibly identical, **not** bit-exact. See [models.md](models.md#int8-quantized-model).

`ModelError` is `#[non_exhaustive]`: `Malformed`, `MissingArray`, `BadSize`, `BadType`,
`UnsupportedVersion`, `ChecksumMismatch { expected, actual }`, `RecordCountMismatch`,
`NoEmbeddedModel`, `Io`.

### Sharing a model across streams / threads

`RnnModel` is `Send + Sync`; wrap it in an `Arc` and give each stream its own
`DenoiseState`:

```rust
use std::sync::Arc;
let model = Arc::new(RnnModel::default().quantized());
let mut a = DenoiseState::new_with_model(model.clone());
let mut b = DenoiseState::new_with_model(model);  // shares the same weights, no copy
```

This is the recommended pattern for many concurrent streams: the weights are
loaded once; only the small per-stream state is duplicated.

### Errors

`ModelError` is returned by `RnnModel::from_bytes`:

| Variant | Meaning |
|---|---|
| `Malformed` | the blob is truncated or not a valid `"DNNw"` record stream |
| `MissingArray(name)` | a required weight array is absent |
| `BadSize(name)` | a weight array has an unexpected length |

It implements `std::error::Error` and `Display`.

## Cargo features

| Feature | Default | Effect |
|---|:--:|---|
| *(none)* | ✓ | the current model, dependency-free |
| `capi` | | export a C ABI matching `rnnoise.h` |
| `legacy-model` | | bundle the original 2018 model as `DenoiseStateV1`; pulls in `realfft` |

```toml
[dependencies]
rnnoise-rs = { version = "0.1", features = ["legacy-model"] }
```

## Legacy (2018) model

With `legacy-model`, the smaller/faster original model is exposed with the same
shape of API:

```rust
# #[cfg(feature = "legacy-model")] {
use rnnoise::{DenoiseStateV1, RnnModelV1, FRAME_SIZE};

let mut st = DenoiseStateV1::new();                 // embedded legacy weights
let v = st.process_frame(&mut out, &inp);

let model = std::sync::Arc::new(RnnModelV1::from_bytes(&blob)?); // custom legacy weights
let mut st = DenoiseStateV1::new_with_model(model);
# Ok::<(), rnnoise::ModelError>(()) }
```

Same 480-sample frames and one-frame delay. Lower quality than the default model,
but faster (see [performance.md](performance.md)). `RnnModelV1::from_bytes` reads
the `nnnoiseless`/old-RNNoise `i8` weight format (not the `"DNNw"` blob).

## C API (`capi` feature)

Build a `cdylib`/`staticlib` with `--features capi`. The exported functions match
the upstream header subset (see [`include/rnnoise.h`](../include/rnnoise.h)):

```c
int   rnnoise_get_frame_size(void);                 // 480
DenoiseState *rnnoise_create(RNNModel *model);      // model may be NULL (default)
void  rnnoise_destroy(DenoiseState *st);
float rnnoise_process_frame(DenoiseState *st, float *out, const float *in);
RNNModel *rnnoise_model_from_buffer(const void *ptr, int len);
RNNModel *rnnoise_model_from_filename(const char *filename);
void  rnnoise_model_free(RNNModel *model);
```

Use the `create`/`process`/`destroy` flow. Unlike the C library, the size-based
`rnnoise_get_size`/`rnnoise_init` pair is intentionally not provided (the Rust
state is opaque). See [building.md](building.md#c-abi) for linking details.

## CLI demo

A drop-in equivalent of upstream `rnnoise_demo` (RAW 16-bit machine-endian mono
PCM @ 48 kHz):

```sh
cargo run --release --bin rnnoise-demo -- noisy.raw denoised.raw
```

It reads/writes RAW `i16` (not WAV) and drops the first output frame, matching
the C demo byte-for-byte (to within ~1 LSB). See [parity.md](parity.md).
