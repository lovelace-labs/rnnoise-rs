//! Tier router for the butterfly stages: the runtime-detected vector tier when the
//! CPU has it, the portable reference otherwise. The decision goes through
//! `simd::tier::resolve`, so `RNNOISE_SIMD_TIER` covers the FFT path as well.
//!
//! SAFETY: the vector tier is called only after `resolve` reports `Sse2` (which comes
//! from `is_x86_feature_detected!`), and every x86-64 CPU satisfies SSE2; the portable
//! reference is the fallback on any other target.

use crate::fft::kiss_fft::Stage;
use crate::fft::scalar;
use crate::fft::Cpx;
#[cfg(target_arch = "x86_64")]
use crate::simd::Tier;

pub(crate) fn process(stages: &[Stage], f: &mut [Cpx]) {
    #[cfg(target_arch = "x86_64")]
    {
        if Tier::resolve(&[Tier::Sse2, Tier::Scalar]) == Tier::Sse2 {
            unsafe { crate::fft::x86_64::process(stages, f) };
            return;
        }
    }
    scalar::process(stages, f);
}
