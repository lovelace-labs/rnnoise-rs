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

#![allow(clippy::needless_range_loop)]

mod celt_lpc;
mod common;
mod denoise;
mod fft;
mod nnet;
mod pitch;
mod weights;

#[cfg(feature = "capi")]
pub mod capi;

pub use denoise::DenoiseState;
pub use weights::{RnnModel, ModelError};

/// Number of samples consumed/produced per [`DenoiseState::process_frame`] call.
pub const FRAME_SIZE: usize = 480;
/// Analysis window length (50% overlap → `2 * FRAME_SIZE`).
pub const WINDOW_SIZE: usize = 2 * FRAME_SIZE;
/// Number of FFT bins kept (`FRAME_SIZE + 1`).
pub const FREQ_SIZE: usize = FRAME_SIZE + 1;

/// Number of perceptual (ERB-like) frequency bands.
pub const NB_BANDS: usize = 32;
/// Length of the neural-network feature vector (`2*NB_BANDS + 1`).
pub const NB_FEATURES: usize = 2 * NB_BANDS + 1;

pub(crate) const PITCH_MIN_PERIOD: usize = 60;
pub(crate) const PITCH_MAX_PERIOD: usize = 768;
pub(crate) const PITCH_FRAME_SIZE: usize = 960;
pub(crate) const PITCH_BUF_SIZE: usize = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE;

/// Band edges in units of FFT bins (50 Hz/bin). Length `NB_BANDS + 2`.
/// "ERB bandwidths going in reverse from 20 kHz" — see `denoise.c`.
pub(crate) const EBAND20MS: [usize; NB_BANDS + 2] = [
    0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60, 68, 77, 87, 98, 110, 124,
    140, 157, 176, 198, 223, 251, 282, 317, 356, 400,
];
