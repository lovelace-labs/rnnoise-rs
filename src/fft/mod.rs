//! Faithful port of the Opus/RNNoise variant of KISS-FFT (`kiss_fft.c`, float
//! configuration). The butterfly operations, twiddle generation and bit-reversal
//! table are replicated exactly so the transform is numerically equivalent to the C
//! reference (`rnn_fft_c`), including the `1/nfft` scaling that the C code applies on
//! the forward transform.
//!
//! RNNoise performs its inverse transform by calling the *forward* FFT on a
//! conjugate-symmetric spectrum (see [`crate::denoise`]), so only the forward
//! transform is needed here.
//!
//! The butterflies have a portable reference ([`scalar`], one expression at a time in
//! `j` order) and vector tiers that keep the same per-element expression order, so
//! they produce identical bits. Tier selection is runtime, not build time.

#[allow(unsafe_code)] // unsafe island: the runtime-detected tier boundary
mod dispatch;
mod kiss_fft;
mod scalar;
mod spectrum;

// Unsafe island; `core::arch` intrinsics are glob-imported.
#[allow(unsafe_code, clippy::wildcard_imports)]
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[allow(unsafe_code, clippy::wildcard_imports)] // unsafe island: tests call the tiers directly
#[cfg(test)]
mod tests;

pub(crate) use kiss_fft::KissFft;
pub(crate) use spectrum::Cpx;
