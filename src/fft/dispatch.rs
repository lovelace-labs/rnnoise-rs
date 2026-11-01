//! Tier router for the butterfly stages: the runtime-detected vector tier when the
//! CPU has it, the portable reference otherwise.

use crate::fft::{plan::Stage, scalar, spectrum::Cpx};

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
