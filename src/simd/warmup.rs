//! Resolve the runtime feature detection **before** the first frame.
//!
//! `std::arch::is_x86_feature_detected!` probes CPUID on the *first* call in the
//! process and caches the answer in a lazy static. Measured on a Zen 3 core: the
//! first call costs ~7.1 µs, every later call ~0.3 ns. So the steady-state frame
//! path is already clean — what a real-time contract cannot accept is that the very
//! first probe happens inside an audio callback, on frame 1.
//!
//! `warmup()` pays that one-time cost at construction instead. It changes no result:
//! the tier ladder picks the same kernel either way, and every tier is bit-identical
//! to the portable reference. See `doc/realtime.md`.

/// Force the cached feature flags for every ISA the tier ladder tests to be resolved
/// now, so the first `process_frame` is not the first probe.
///
/// Calling this is never required for correctness — it only moves a one-time cost out
/// of the frame path. [`crate::DenoiseState::new_with_model`] calls it, so a caller
/// that constructs its state before entering the real-time loop does not need to call
/// it at all. A caller that builds `DenoiseState` inside the callback (or that wants
/// the whole startup cost measurable up front) can call it at startup.
#[cfg(target_arch = "x86_64")]
pub fn warmup() {
    // Exactly the features `simd::x86_64::select` and `fft::dispatch` test at runtime,
    // top-down. `black_box` keeps the calls: their only effect is the one-time cache fill.
    std::hint::black_box(std::arch::is_x86_feature_detected!("avx512f"));
    std::hint::black_box(std::arch::is_x86_feature_detected!("avx512vnni"));
    std::hint::black_box(std::arch::is_x86_feature_detected!("avx2"));
    std::hint::black_box(std::arch::is_x86_feature_detected!("sse4.1"));
    std::hint::black_box(std::arch::is_x86_feature_detected!("sse2"));
}

/// Force the cached feature flags for the `dotprod` tier to be resolved now.
#[cfg(target_arch = "aarch64")]
pub fn warmup() {
    std::hint::black_box(std::arch::is_aarch64_feature_detected!("dotprod"));
}

/// Nothing to resolve on targets without a runtime-detected tier.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn warmup() {}
