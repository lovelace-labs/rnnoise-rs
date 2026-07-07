# Models & weights

`rnnoise-rs` ships two models and can load custom ones. This document covers what
is bundled, the on-disk weight format, how to load your own weights, and the
optional int8 and legacy paths.

## What's bundled

| File | Used by | Size | Format |
|---|---|--:|---|
| [`models/rnnoise_default.bin`](../models/rnnoise_default.bin) | the current model (default) | 11.3 MB | `"DNNw"` blob, float weights |
| [`models/rnnoise_legacy.rnn`](../models/rnnoise_legacy.rnn) | `legacy-model` feature | 87 KB | old-RNNoise `i8` layer stream |

Both are embedded at compile time with `include_bytes!`, so the library is
self-contained — no files to ship, no download at runtime. The legacy blob is
only embedded when the `legacy-model` feature is on.

> **crates.io note:** the default blob (11.3 MB) exceeds the default crates.io
> 10 MB crate-size limit. Publishing would need either the size limit raised, or
> shipping a smaller (int8 / "little") blob. See [Future work](#future-work).

## The current model

The default model is the official Xiph RNNoise model (checksum-verified against
`model_version`). Architecture and exact dimensions are in
[architecture.md](architecture.md#the-neural-network-in-detail). The bundled blob
is a **float-only slim** version: the upstream model ships both int8 and float
weights for the quantizable layers, and the C reference uses the float ones, so
embedding just the float arrays reproduces the reference exactly while keeping the
blob smaller. The arrays present (29 of them):

```
conv1_weights_float, conv1_bias
conv2_weights_float, conv2_bias
gru{1,2,3}_input_weights_float, _input_weights_idx, _input_bias
gru{1,2,3}_recurrent_weights_float, _recurrent_weights_idx,
          _recurrent_weights_diag, _recurrent_bias
dense_out_weights_float, dense_out_bias
vad_dense_weights_float, vad_dense_bias
```

The GRU matrices use a sparse 8×4-block layout indexed by the `*_weights_idx`
arrays. In the regular model every block is present (it is dense), but the loader
and GEMV honour the block/index structure regardless, so the sparse "little"
model would also load.

## The `"DNNw"` blob format

This is upstream RNNoise's own "machine-endian" weight format (produced by
`dump_weights_blob` / `write_weights.c`), which `RnnModel::from_bytes` parses —
so a blob exported from upstream loads directly, and vice-versa.

A blob is a sequence of records, each:

```
┌──────────────────────────── 64-byte header (WeightHead) ───────────────────────────┐
│ char head[4] = "DNNw"                                                                │
│ i32  version  = 0                                                                    │
│ i32  type     (0 = float, 1 = int32, 2 = qweight, 3 = int8)                          │
│ i32  size     (payload bytes)                                                        │
│ i32  block_size = ceil(size / 64) * 64   (payload padded to 64-byte alignment)      │
│ char name[44] (NUL-terminated)                                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
  payload[size]  followed by (block_size - size) zero pad bytes
```

All integers are little-endian (the practical meaning of upstream's "machine
endian"). `rnnoise-rs` reads little-endian; big-endian hosts are not currently
supported (see [Future work](#future-work)). Float arrays are raw `f32`; index
arrays are raw `i32`.

The loader ([`weights::parse_weights`](../src/weights.rs)) walks the records into
a name→array map, then `init`-style code wires the named arrays into the ten
`LinearLayer`s, validating every length (dense layers `nb_inputs*nb_outputs`,
sparse layers `32 * total_blocks` from the index structure, biases/diag
`nb_outputs`).

### Generating the default blob

The bundled `models/rnnoise_default.bin` was generated from the official model's
C source with [`scripts/gen_blob.c`](../scripts/gen_blob.c): it `#include`s the
upstream `rnnoise_data.c`, then writes the subset of float arrays the float
compute path needs, in the `"DNNw"` record format. To reproduce, build it against
an extracted upstream model tree and run it (the script lists the exact arrays).

## Loading a custom model

```rust
use std::sync::Arc;
use rnnoise::{DenoiseState, RnnModel};

let blob = std::fs::read("weights_blob.bin")?;     // upstream "DNNw" export
let model = Arc::new(RnnModel::from_bytes(&blob)?);
let mut st = DenoiseState::with_model(model);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The model's layer dimensions are fixed by this port (they match the shipped
architecture); a blob with different dimensions will fail validation with
`ModelError::BadSize`. To run a differently-shaped model you'd adjust the
dimensions in `weights::RnnModel::from_bytes`.

## int8 quantized model

`RnnModel::quantized()` converts conv2 and the GRU matrices to int8 (conv1 and the
output heads stay float, mirroring upstream's choice):

```rust
use std::sync::Arc;
let model = Arc::new(rnnoise::RnnModel::default().quantized());
let mut st = rnnoise::DenoiseState::with_model(model);
```

- **Layout:** each quantized layer becomes a transposed, dense `[output][input]`
  int8 matrix plus a per-output `f32` scale. The scale folds in both the weight
  scale (`max|w|/127`) and the activation scale (`1/127`), so inference is
  `out[o] = scale[o] · (q_w[o] · q_x)` with an int32 dot product.
- **Runtime:** activations (all `tanh`/`sigmoid`/GRU outputs in `[-1, 1]`) are
  quantized to int8, and each output is computed with a NEON `sdot` 4-MACs-per-
  lane dot product (inline asm; the `vdotq_s32` intrinsic is still nightly-only),
  with a portable scalar fallback that gives the identical integer result.
- **Footprint:** ~11.5 MB → ~2.9 MB of weights.
- **Speed:** ~3× faster than the float path on Apple Silicon (the NN itself ~4×).
  See [performance.md](performance.md).
- **Accuracy:** not bit-exact — `rel. energy 4.8e-6` vs the float model on the
  test signal (audibly identical). That's why it's opt-in and the float model is
  the default.

Why int8 and not f16/float-stored? On this workload it's *instruction*-bound, not
bandwidth-bound; `sdot` cuts the instruction count ~4×. A naive int8 path
(dequantize to f32 in the loop) is actually *slower*; storing f32 weights is
slower still (4× cache traffic). See [performance.md](performance.md).

## Legacy (2018) model

The `legacy-model` feature bundles the original RNNoise model (the one
`nnnoiseless` uses) and exposes it as `DenoiseStateV1` / `RnnModelV1`.

- **Format:** a flat stream of `i8` layer records (`nnnoiseless`'s format):
  for each layer `nb_inputs, nb_neurons, activation`, then the weights and bias.
  Layers in order: input-dense, vad-gru, noise-gru, denoise-gru, denoise-output,
  vad-output. Activations: `0=tanh, 1=sigmoid, 2=relu`.
- **`RnnModelV1::from_bytes(&[u8])`** loads this format (validating the layer
  wiring); the bundled `models/rnnoise_legacy.rnn` is the default.
- **Speed:** *faster than nnnoiseless* — it uses `realfft` (rustfft) for the FFT,
  4-wide pitch kernels, and `sdot` int8 GRU input matmuls. See
  [performance.md](performance.md).
- **Quality:** lower than the current model (it's the old, smaller network).

## Future work

- Ship a pre-quantized int8 blob (~2.9 MB) and/or the sparse "little" model behind
  a feature, so the crate fits the crates.io 10 MB limit.
- Big-endian support in the blob loader (currently little-endian only).
- x86 VNNI kernel for the int8 path (the scalar fallback already runs everywhere).
