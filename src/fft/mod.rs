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

mod dispatch;
mod plan;
mod scalar;
mod spectrum;

#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(test)]
mod tests;

pub(crate) use plan::KissFft;
pub(crate) use spectrum::Cpx;
