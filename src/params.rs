//! Model geometry: the frame/window/bin constants and the precomputed band-edge
//! table. Every value here is part of the parity contract with the C reference
//! (`denoise.c`, `rnnoise_tables.c`) — the band edges are the upstream table,
//! reproduced as the nearest `f32`/`usize` values.

/// Number of samples consumed/produced per [`crate::DenoiseState::process_frame`] call.
pub const FRAME_SIZE: usize = 480;
/// Analysis window length (50 % overlap → `2 * FRAME_SIZE`).
pub const WINDOW_SIZE: usize = 2 * FRAME_SIZE;
/// Number of FFT bins kept (`FRAME_SIZE + 1`).
pub const FREQ_SIZE: usize = FRAME_SIZE + 1;

/// Number of perceptual (ERB-like) frequency bands.
pub const NB_BANDS: usize = 32;
/// Length of the neural-network feature vector (`2 * NB_BANDS + 1`).
pub const NB_FEATURES: usize = 2 * NB_BANDS + 1;

pub(crate) const PITCH_MIN_PERIOD: usize = 60;
pub(crate) const PITCH_MAX_PERIOD: usize = 768;
pub(crate) const PITCH_FRAME_SIZE: usize = 960;
pub(crate) const PITCH_BUF_SIZE: usize = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE;

/// Largest input width of a quantized layer. `RnnModel::quantized()` quantizes conv2 and the GRU
/// matrices, which are all 384 wide, so every int8 row in the model is 384 bytes — 6 AVX-512
/// chunks or 24 SSE2 chunks. It is also the size of the activation buffer in `nnet`.
pub(crate) const MAX_Q8_INPUTS: usize = 384;

/// Band edges in units of FFT bins (50 Hz/bin). Length `NB_BANDS + 2`.
/// "ERB bandwidths going in reverse from 20 kHz" — see `denoise.c`.
pub(crate) const EBAND20MS: [usize; NB_BANDS + 2] = [
    0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60, 68, 77, 87, 98, 110, 124,
    140, 157, 176, 198, 223, 251, 282, 317, 356, 400,
];
