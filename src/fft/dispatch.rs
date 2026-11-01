//! Tier router for the butterfly stages: the runtime-detected vector tier when the
//! CPU has it, the portable reference otherwise.
//!
//! SAFETY: the vector tier is called only after `is_x86_feature_detected!("sse2")`,
//! which every x86-64 CPU satisfies; the portable reference is the fallback on any
//! other target.

use crate::fft::kiss_fft::Stage;
use crate::fft::scalar;
use crate::fft::Cpx;

pub(crate) fn process(stages: &[Stage], f: &mut [Cpx]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            unsafe { crate::fft::x86_64::process(stages, f) };
            return;
        }
    }
    scalar::process(stages, f);
}
