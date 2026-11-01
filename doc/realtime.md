# Real-time contract

`rnnoise-rs` is a real-time DSP library: `process_frame` is normally called from an
audio callback, sometimes under real-time scheduling. This page states **what the frame
path guarantees**, **how each part of that is enforced**, and — just as important —
**what it does not guarantee**.

Short version: after construction, `process_frame` allocates nothing, takes no lock,
does no I/O, logs nothing, and does a fixed amount of work per frame. It is *bounded
work*, not a WCET proof.

## What is guaranteed

| Property | Where it holds | How it is enforced |
|---|---|---|
| **No heap allocation** in the frame path | every buffer inside `process_frame` and its helpers is a fixed-size array; the state itself is fixed-size arrays; the model is parsed once at construction | `tests/real_time_no_alloc.rs` (counting `GlobalAlloc`), `scripts/realtime_gate.py` (source gate) |
| **No locks** | nothing in the frame path touches a `Mutex`, `RwLock` or semaphore; the shared tables and the shared model are resolved at construction, so the frame path holds no lazy state; `Arc<RnnModel>` is immutable after construction | `scripts/realtime_gate.py`; review |
| **No I/O, no logging** | nothing reads the filesystem, the environment, or a clock, and nothing prints, in `src/` except `src/bin/` (the CLI) | `scripts/realtime_gate.py` |
| **Bounded work per frame** | every loop in the DSP path is fixed-size and bounded by the model geometry (`FRAME_SIZE`, `FREQ_SIZE`, `NB_BANDS`, `PITCH_MAX_PERIOD`, the layer shapes). There is no unbounded loop, no recursion, no variable iteration count. The worst case is the **non-silent** frame, because silence skips the network | `doc/architecture.md`; `scripts/realtime_gate.py`; the bench |
| **No data-dependent slow path for realistic input** | subnormals cannot be produced at the documented amplitudes (see below); the `1e-2`/`1e-3`/`1e-8`/`0.001` guards keep a zero from becoming a division | `tests/real_time_denormals.rs` |
| No threads, no `catch_unwind`, no async, nothing before `main()` | `scripts/realtime_gate.py` + review | gate + review rule |

The guarantee is scoped to a **well-formed call**: `input` and `output` are at least
`FRAME_SIZE` long. The only panic reachable from `process_frame` is that violated
invariant, and a panic path can format a message (which allocates) and unwind — so the
"no allocation" claim is about the normal path, not about the failure path.

## Cold start, and why construction is the boundary

Everything one-time happens in `DenoiseState::new_with_model` (or `DenoiseStateV1::new_with_model`):

- the shared tables (`Common`: window, DCT matrix, band weights, the FFT plan),
- the weight parse,
- the **CPUID probe** for the runtime tier ladder — `std::arch::is_x86_feature_detected!`
  probes once per process and caches the answer. Measured: **first call 7.1 µs, every
  later call 0.3 ns**, so the steady-state path was already clean; what was not acceptable
  for a real-time callback was that the *first* probe could happen inside frame 1.
  `rnnoise::warmup()` resolves the flags, and `new_with_model` calls it, so nothing
  first-use-lazy is left reachable from `process_frame`.

`warmup()` changes no result: whichever tier is picked, the kernels are bit-identical to
the portable reference (`doc/parity.md`, `src/simd/tests.rs`).

Measured cold start (fresh process, mean of three runs on this box): **construction
≈ 21.3 ms**, first frame ≈ 1.06 ms. That is startup, not the frame path — but it is the
number you need if you must be real-time from the first callback: build the state (and call
`warmup()`) before entering the callback. Cold start is the noisiest number here
(14–22 ms between runs); the frame path is not.

## Denormals

A `f32` can only be subnormal below the smallest normal, `1.1754944e-38`. In this pipeline
subnormals can only be *produced* by a multiplication or a square, so the boundary is an
input amplitude of `√1.1754944e-38 ≈ 1.084e-19` — 19 orders of magnitude below the LSB of
16-bit PCM, i.e. not audio. On the test signal, no output sample is subnormal
(`tests/real_time_denormals.rs`); the smallest non-zero input sample is ~1.0.

The measured cost of a subnormal path, for comparison (300-element band pass, same shape
as `compute_band_energy`): **160.9 ns → 247.9 ns**, i.e. ~1.54×. So a denormal-heavy frame
costs hundreds of nanoseconds — not enough to break a 10 ms budget, but enough that a
"no data-dependent slow path" claim has to be stated as a boundary rather than a law.

The crate deliberately does **not** set FTZ/DAZ (x86 `MXCSR`, aarch64 `FPCR`). Setting it
is a per-thread process-wide state change, and for exactly those pathological inputs it
would change results — which is outside the bit-exact parity contract. A caller that wants
flushing (e.g. in the caller's capture path) can set it itself; that is the caller's policy call,
not the crate's.

## Measured distribution

`make bench-dist` (`cargo bench --all-features --bench real_time_distribution`) on an AMD
Ryzen 9 5900X (Zen 3), rustc 1.99.1, release profile (`lto = true`, `codegen-units = 1`),
single thread, no real-time scheduling, three runs of 20 000 frames each after a 200-frame
warm-up, each frame timed individually (`Instant`, ~30–50 ns of overhead included); the
table is the mean of the three runs. Frame budget: 10 ms.

| path | mean | p50 | p99 | p99.9 | max | headroom (mean / max) |
|---|--:|--:|--:|--:|--:|--:|
| float (default) | 179.0 µs | 171.1 | 288.8 | 777.5 | **1525.8 µs** | 55.9× / 6.6× |
| int8 quantized | 107.2 µs | 102.3 | 175.3 | 304.8 | **1876.7 µs** | 93.3× / 5.3× |
| legacy (2018) | 35.6 µs | 33.9 | 54.3 | 88.5 | **1013.3 µs** | 280.9× / 9.9× |
| float, front-end only (silence) | 29.6 µs | 28.5 | 57.6 | 85.3 | 785.1 µs | 337.8× / 12.7× |

These are ~4 % above the pre-`Tier::resolve` numbers for the same code path — the cost of the
per-call ladder walk dev added, and still ~56× headroom on the mean.

The max/p99.9 tail is **OS noise** (scheduler, IRQ, page faults), not algorithmic work:
the per-frame work is fixed, so the only way to see a 1 ms frame is to be preempted. On
a pinned core under `SCHED_FIFO` the tail is much tighter; if you need a number for a
specific deployment, measure there — the bench reports percentiles precisely so that
measurement is comparable.

Allocation counts over the measured window: **0 allocations over 20 000 frames**, ~66–71
allocations total during construction. The one exception the test measures separately is
one-time *runtime* bookkeeping on the measuring thread (lazy TLS/CRT init) — Windows CI showed
exactly 1 block on the very first window. That is not the frame path: the same call in the
next window is 0, so the test asserts 0 per frame and tolerates 1 for the first call on a
fresh thread.

## What is not guaranteed

- **No WCET.** Rust has no worst-case execution-time model, the optimizer can change the
  shape of the loop, page faults and scheduler preemption are outside the library, and the
  libm calls used by the feature path (`log10`, the tanh/sigmoid approximations) are
  data-dependent inside their own bounds. The honest claim is: fixed-size work per frame,
  measured percentiles, and no allocation/lock/I/O in the steady state.
- **Not `Sync` per stream.** One `DenoiseState` is one stream; process one stream on one
  thread. Sharing a `RnnModel` across threads is fine (it is immutable after construction),
  sharing a `DenoiseState` is not.
- **The failure path may allocate**, as stated above.
- **`capi`**: `rnnoise_create` allocates (it must — it returns an opaque handle), and
  `rnnoise_process_frame` inherits the frame-path guarantee.

## Using it from a real-time callback

```rust
use rnnoise::{DenoiseState, FRAME_SIZE, warmup};

// Before entering the callback: build state, resolve the CPUID cache.
warmup();                       // also done by DenoiseState::new_with_model
let mut st = DenoiseState::new();
let mut input = [0.0f32; FRAME_SIZE];
let mut output = [0.0f32; FRAME_SIZE];

fn callback(&mut self) {
    // no allocation, no lock, no I/O, no logging in here:
    let _vad = st.process_frame(&mut output, &input);
}
```

Call `warmup()` if you want the one-time probe attributed to startup rather than to the
first frame; call it before `DenoiseState::new()` if you construct the state inside the
callback.

## See also

- [`performance.md`](performance.md) — throughput and the SIMD tiers.
- [`parity.md`](parity.md) — why the tier choice cannot change output.
- `scripts/realtime_gate.py` — the source-level gate, part of `make check`.
- `benches/real_time_distribution.rs` (`make bench-dist`) — the per-frame distribution; the
  criterion bench in `benches/process.rs` stays the CI baseline harness.
