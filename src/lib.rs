//! `rnnoise-rs` — an idiomatic, dependency-free Rust port of Xiph
//! [RNNoise](https://gitlab.xiph.org/xiph/rnnoise), tracking the **current**
//! (2024) architecture: a 32-band DSP front-end feeding a 2×Conv1D + 3×GRU
//! recurrent network that predicts per-band suppression gains.
//!
//! This is a faithful port of `rnnoise_process_frame`: it operates on
//! **480-sample frames of 48 kHz mono `f32` audio** and is numerically
//! equivalent to the upstream C library (see `tests/parity.rs`).
//!
//! # Example
//! ```no_run
//! use rnnoise::DenoiseState;
//!
//! let mut st = DenoiseState::new();
//! let input = [0.0f32; rnnoise::FRAME_SIZE];
//! let mut output = [0.0f32; rnnoise::FRAME_SIZE];
//! let vad_probability = st.process_frame(&mut output, &input);
//! ```
//!
//! Sample amplitudes use the same convention as upstream: roughly the range of
//! 16-bit PCM (i.e. `i16` values cast to `f32`), not `[-1, 1]`.

mod celt_lpc;
mod common;
mod denoise;
mod fft;
mod nnet;
mod params;
mod pitch;
#[allow(unsafe_code)] // unsafe island: core::arch intrinsics are unsafe functions
mod simd;
mod weights;

#[cfg(feature = "capi")]
#[allow(unsafe_code)] // unsafe island: pointers cross the FFI boundary
pub mod capi;

/// The original (2018) RNNoise model — smaller and faster, lower quality.
/// Enable with the `legacy-model` feature.
#[cfg(feature = "legacy-model")]
pub mod legacy;

pub use denoise::DenoiseState;
pub use params::{FRAME_SIZE, FREQ_SIZE, NB_BANDS, NB_FEATURES, WINDOW_SIZE};
pub use weights::{ModelError, RnnModel};

/// Resolve the runtime CPU feature detection up front, so the first `process_frame`
/// is not the first CPUID probe (`doc/realtime.md`).
pub use simd::warmup;

#[cfg(feature = "legacy-model")]
pub use legacy::{DenoiseStateV1, RnnModelV1};
