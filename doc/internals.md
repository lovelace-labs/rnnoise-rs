# Internals (for contributors)

A module-by-module tour of the implementation, with the subtle, parity-critical
details called out. Read [architecture.md](architecture.md) first for the big
picture. The guiding principle throughout: **match the C `_c` (scalar) reference
operation-for-operation**, including its float/double promotions and its quirks,
so the output is numerically equivalent (see [parity.md](parity.md)).

## `fft` — KISS-FFT and RealFft

[`src/fft.rs`](../src/fft.rs)

`KissFft` is a faithful port of the Opus/RNNoise variant of KISS-FFT (float
config): `kf_factor` (radix 4/2/3/5, reversed so radix-4 lands last), recursive
`compute_bitrev_table`, `compute_twiddles` (`exp(-2πi k/n)` in `f64` → `f32`),
and the `bfly2/3/4/5` butterflies, all transcribed with the same arithmetic and
ordering as the C macros (`C_MUL`, `C_ADD`, `HALF_OF`, …). 960 factors as
`5·3·4·4·4`.

- `forward(fin, fout)` matches `rnn_fft_c`: it scales the input by `1/nfft`,
  applies the bit-reversal permutation, then runs the in-place butterflies.
- `forward_scaled(fin, fout, scale)` is the same with a caller-chosen input
  scale. The current model's `forward_transform` uses `1/nfft`; the inverse is
  done (as upstream does) by calling the *forward* FFT on a conjugate-symmetric
  spectrum.
- `RealFft` (used only by the legacy model) is the standard "pack reals into a
  half-length complex FFT, then split" real transform; `forward` returns the
  unnormalized `DFT_n`, `inverse` the unnormalized inverse (`n×` the true
  inverse). It is validated against a naive DFT in unit tests.

Why two FFTs? The current model must stay **bit-exact** with the C reference,
which uses this exact KISS complex FFT — so it cannot be swapped. The legacy model
is not bit-exact (it tolerates ~1e-5), so it uses the faster `realfft` (rustfft).

## `common` — tables, bands, biquad, transforms

[`src/common.rs`](../src/common.rs)

Built once into a `Common` (window, DCT matrix, FFT plan) behind a `OnceLock`.

- **Window / DCT tables** are generated from the exact upstream formulas
  (`rnnoise_tables.c`): `half_window[i] = sin(½π · sin²(½π(i+½)/FRAME_SIZE))`;
  `dct_table[i·32+j] = cos((i+½)·j·π/32)` (×`√½` when `j=0`), all in `f64` then
  cast to `f32`, so they match the embedded C tables.
- **`dct`** multiplies by `DCT_SCALE = √(2/22)` — note the `22`, a deliberate
  hold-over from the old 22-band model that the trained weights expect. The final
  multiply is done in `f64` (`(sum as f64 * DCT_SCALE) as f32`) like the C.
- **`compute_band_energy` / `compute_band_corr`** use a 34-entry scratch with the
  triangular `(1-frac)/frac` split and the `*2/3` weighting at the two end bands,
  exactly as `denoise.c` (the `(a*2)/3` association is preserved).
- **`interp_band_gain`** reproduces a subtle upstream behaviour: the C
  `memset(g, 0, FREQ_SIZE)` only zeroes `FREQ_SIZE` *bytes* (a bug), but every
  caller pre-zeroes its buffer and the loops only write bins `0..400`, so the net
  effect is "zero the whole buffer, write `0..400`". We zero `g[..FREQ_SIZE]`
  fully and write `0..400` — identical result, no garbage.
- **`biquad`** keeps the filter state update in `f64` (`mem[0] = mem[1] + (b0·x −
  a0·y)`), matching the C high-pass; the DC-removal coefficients are
  `b = [-2, 1]`, `a = [-1.99599, 0.99600]`.

## `celt_lpc` — LPC and correlation kernels

[`src/celt_lpc.rs`](../src/celt_lpc.rs)

In the float build all the CELT fixed-point macros (`SHR32`, `MULT16_16_Q15`,
`ROUND16`, …) are identities or plain float ops, so these are straightforward
float routines:

- `lpc` — Levinson-Durbin (the `<<3`/`>>3` shifts cancel to identities in float).
- `autocorr`, `fir5` (order-5 FIR applied in place for pitch whitening).
- `inner_prod` / `dual_inner_prod` — **four-accumulator** dot products: the loop
  uses 4 independent partial sums (combined at the end) so it vectorizes. This
  reorders the float additions vs a strict left-to-right sum, but the divergence
  is sub-LSB and the new-model parity test confirms it changes nothing
  observable.
- `pitch_xcorr` — computes four lags at once, advancing through `x` while keeping
  a small window of `y` in registers (each `x` load is reused four times). Each
  `xcorr[i]` still accumulates in `j` order. This is the hottest kernel in the
  pitch search.

## `pitch` — pitch analysis

[`src/pitch.rs`](../src/pitch.rs)

- `pitch_downsample` — 2× decimate, autocorrelation, lag-windowing, LPC(4), then
  a whitening FIR.
- `pitch_search` — coarse search on 4×-decimated data (`pitch_xcorr` +
  `find_best_pitch`), a finer search on 2×-decimated data near the two best
  candidates, then pseudo-interpolation.
  - `find_best_pitch` maximizes `xcorr²/‖y‖²`. The upstream `1e-12` pre-scale is
    omitted: it's a uniform factor on `num`, and the comparison
    `num·den₂ > num₂·den` is scale-invariant, so the chosen lag is identical.
- `remove_doubling` — detects pitch period doubling/halving. The C does `x +=
  maxperiod` and then indexes negatively; we emulate that with an explicit base
  offset `xb` so `x[k]`/`x[-k]` become `x[xb±k]`. The threshold ladder
  (`if t1<3·min … else if t1<2·min …`) is ported verbatim including the branch
  that is logically unreachable (kept for exactness). `compute_pitch_gain` is
  `xy / √(1 + xx·yy)` with the `√` and division in `f64`, as in C.

## `nnet` — the neural network

[`src/nnet.rs`](../src/nnet.rs)

- **Activations** `tanh_approx` / `sigmoid_approx` are the exact `vec.h`
  rational-polynomial approximations (coefficients `N0..D2`), so they match the C
  `_c` path bit-for-bit. The clamp is `min(1).max(-1)` mirroring `MAX(-1,
  MIN(1, x))`.
- **`LinearLayer`** holds either `Weights::Float` or `Weights::Q8`:
  - *Dense float* (`dense_sgemv`) — accumulated as a sequence of SAXPYs
    (`j` outer, `i` inner). This streams the matrix sequentially and
    auto-vectorizes, while preserving the exact per-`out[i]` accumulation order,
    so it is **bit-identical** to the C `sgemv` and to a strided version.
  - *Sparse float 8×4* (`sparse_sgemv8x4`) — the GRU matrices; bounds-check-free
    8-wide row updates over the indexed blocks.
  - *int8* (`dense_q8`) — quantize activations, then per-output `sdot` (or scalar)
    int32 dot, scaled. See [models.md](models.md#int8-quantized-model).
  - A `diag` term is added for GRU recurrent matrices (`3M == N`).
- **`sdot`** is a single `sdot v.4s, a.16b, b.16b` via stable inline `asm!`
  (the `vdotq_s32` intrinsic is nightly-only). `dense_q8_neon` loops it over
  16-element chunks; `dense_q8_scalar` is the portable fallback with the identical
  integer result (unit-tested by `sdot_matches_scalar`). `i8_dot` (gated to
  `legacy-model`) is the single-vector version the legacy GRUs use.
- **`compute_conv1d`** shifts `mem` + `input` through the kernel window (the
  layer's `nb_inputs` is `3 × in_size`; `mem` holds the previous 2 frames), then
  the linear layer + activation, and updates `mem`.
- **`compute_gru`** implements the update/reset/candidate GRU (the recurrent
  candidate uses the reset-gated state). `compute_rnn` wires conv1→conv2→
  gru1→gru2→gru3, builds the 1536-wide concatenation, and runs the two output
  heads. `RnnState` holds the conv kernel memories and the three GRU states.

## `weights` — model loading

[`src/weights.rs`](../src/weights.rs)

`parse_weights` walks the `"DNNw"` records into a `name → (type, bytes)` map (with
header validation); `f32_array`/`i32_array` extract typed arrays (little-endian);
`linear` builds each layer, validating sizes (dense `m·n`, sparse `32·blocks` from
`sparse_total_blocks`, bias/diag `n`). `RnnModel::from_bytes` wires the ten named
layers; `RnnModel::quantized` produces the int8 variant. The default model is
`include_bytes!`-embedded and parsed lazily behind an `Arc` (see `denoise`).

## `denoise` — the pipeline

[`src/denoise.rs`](../src/denoise.rs)

`DenoiseState::process_frame` orchestrates the per-frame flow from
[architecture.md](architecture.md). Notable details:

- **Double-precision promotions** are replicated to match the C exactly, e.g.
  `log10(1e-2 + Ex[i])`, the `follow - 1.5` / `logMax - 7` cepstral clamps,
  `Exp[i] / √(0.001 + Ex[i]·Ep[i])`, the `pitch_filter` `SQUARE(...)` ratio with
  its `.001` (double) denominator, and the gain-smoothing
  `g[i]·(dEx+1e-3)/(Ex+1e-3)` — all done in `f64` where the C uses `double`
  literals, then cast back to `f32`.
- **Disjoint-field borrows:** `frame_synthesis` and `pitch_filter` are free
  functions taking the specific fields they touch (e.g. `&mut delayed_x` plus
  `&delayed_p`), so the borrow checker accepts the simultaneous access without
  copies.
- **The one-frame delay** is the `delayed_*` set, rotated at the end of every
  frame; gains are applied to the delayed spectrum (see architecture).
- **Gain smoothing:** `g[i] = max(g[i], 0.6·lastg[i])` (RT60 ≈ 135 ms decay cap)
  and `lastg[i] = min(1, g[i]·(delayed_Ex+1e-3)/(Ex+1e-3))` (cross-frame energy
  compensation), matching `denoise.c`.

## `legacy` — the 2018 model

[`src/legacy.rs`](../src/legacy.rs) (feature `legacy-model`)

A self-contained second pipeline: 22 bands (`EBAND_5MS`), a 42-D feature vector
built from cepstra + Δ/ΔΔ history (`cepstral_mem`) + spectral variability,
table-based `tansig_approx` activations, and the input-dense + 3-GRU network in
its own `i8` weight format. It reuses this crate's `biquad` and `pitch`/`celt_lpc`
kernels, uses `realfft` for the FFT, and accelerates the GRU input matmuls with
`nnet::i8_dot` (`sdot`) over dynamically-quantized activations. See
[performance.md](performance.md) for why each of those choices was made.

## Tests

- `tests/parity.rs` — end-to-end vs the C reference (`matches_c_reference`) and
  int8 accuracy (`quantized_accuracy`).
- `tests/legacy.rs` — legacy output vs the old model (`legacy_matches_old_model`).
- `tests/api.rs` — determinism, custom-model round-trips, silence, error handling.
- unit tests in `fft` (DFT cross-check, factorization, `RealFft` round-trip) and
  `nnet` (activation bounds, `sdot` == scalar).

See [parity.md](parity.md) for the oracle and tolerances, and
[building.md](building.md) for how to run everything.
