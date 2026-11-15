//! The named tiers of the ladder, what the running CPU has, and the override CI uses
//! so every kernel path is actually executed on a runner.
//!
//! [`Tier::resolve`] is the single decision point: given a kernel's `ladder` (best →
//! worst, ending at `Scalar`), it returns the tier to run. Without an override that is
//! exactly the top-down ladder `select` used before, so production behaviour is
//! unchanged.
//!
//! The override is `cfg(test)`-only: in a normal build `requested()` is a constant
//! `None` and the thread-local is not compiled at all, so the real-time path pays
//! nothing for it — no env lookup, no allocation, nothing per call. CI sets
//! `RNNOISE_SIMD_TIER`, and `simd::coverage` then runs the pipeline once per tier the
//! runner reports.
//!
//! A tier is never forced onto a CPU that lacks the ISA: `resolve` only hands back a
//! tier that `is_available` confirms, so coverage is bounded by the runner — x86-64
//! gives v1–v3, aarch64 gives `dotprod`, v4 only on a machine with AVX-512F+VNNI. The
//! pure-Rust VNNI identity in `simd::tests` is the fallback for v4 elsewhere.

#[cfg(test)]
use core::cell::Cell;

/// One ISA level of the ladder. `Scalar` is the portable reference that every tier must
/// match bit for bit (`simd/mod.rs`); the other variants exist only on the architecture
/// that has them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tier {
    /// Portable reference, no vector ISA.
    Scalar,
    /// x86-64 v1: SSE2.
    #[cfg(target_arch = "x86_64")]
    Sse2,
    /// x86-64 v2: SSE4.1 (byte sign-widen).
    #[cfg(target_arch = "x86_64")]
    Sse41,
    /// x86-64 v3: AVX2.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// x86-64 v4: AVX-512F + AVX-512 VNNI.
    #[cfg(target_arch = "x86_64")]
    Vnni,
    /// AArch64: `dotprod`.
    #[cfg(target_arch = "aarch64")]
    Dotprod,
}

impl Tier {
    /// Ladder rank, used to cap a forced tier to the best one a kernel actually has.
    #[must_use]
    pub(crate) fn rank(self) -> u8 {
        match self {
            Tier::Scalar => 0,
            #[cfg(target_arch = "x86_64")]
            Tier::Sse2 => 1,
            #[cfg(target_arch = "x86_64")]
            Tier::Sse41 => 2,
            #[cfg(target_arch = "x86_64")]
            Tier::Avx2 => 3,
            #[cfg(target_arch = "x86_64")]
            Tier::Vnni => 4,
            #[cfg(target_arch = "aarch64")]
            Tier::Dotprod => 1,
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            _ => 0,
        }
    }

    /// Name as it appears in the CI coverage table and in `RNNOISE_SIMD_TIER`.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Tier::Sse2 => "sse2",
            #[cfg(target_arch = "x86_64")]
            Tier::Sse41 => "sse4.1",
            #[cfg(target_arch = "x86_64")]
            Tier::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Tier::Vnni => "vnni",
            #[cfg(target_arch = "aarch64")]
            Tier::Dotprod => "dotprod",
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            _ => "scalar",
        }
    }

    /// Parse `RNNOISE_SIMD_TIER`; unknown names are ignored, so the runner falls back
    /// to plain detection rather than failing the job.
    #[must_use]
    #[cfg(any(test, feature = "simd-tier-override"))]
    pub(crate) fn parse(value: &str) -> Option<Tier> {
        match value.trim().to_ascii_lowercase().as_str() {
            "scalar" | "c" => Some(Tier::Scalar),
            #[cfg(target_arch = "x86_64")]
            "sse2" | "v1" => Some(Tier::Sse2),
            #[cfg(target_arch = "x86_64")]
            "sse4.1" | "sse41" | "ssse3" | "v2" => Some(Tier::Sse41),
            #[cfg(target_arch = "x86_64")]
            "avx2" | "v3" => Some(Tier::Avx2),
            #[cfg(target_arch = "x86_64")]
            "vnni" | "avx512vnni" | "v4" => Some(Tier::Vnni),
            #[cfg(target_arch = "aarch64")]
            "dotprod" | "neon" => Some(Tier::Dotprod),
            _ => None,
        }
    }

    /// Whether this CPU has the ISA the tier needs.
    #[must_use]
    pub(crate) fn is_available(self) -> bool {
        match self {
            Tier::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Tier::Sse2 => std::arch::is_x86_feature_detected!("sse2"),
            #[cfg(target_arch = "x86_64")]
            Tier::Sse41 => std::arch::is_x86_feature_detected!("sse4.1"),
            #[cfg(target_arch = "x86_64")]
            Tier::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Tier::Vnni => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512vnni")
            }
            #[cfg(target_arch = "aarch64")]
            Tier::Dotprod => std::arch::is_aarch64_feature_detected!("dotprod"),
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            _ => true,
        }
    }

    /// Tier to run for a kernel whose ladder is `ladder` (best → worst, ending at
    /// `Scalar`): the forced tier capped at the best tier this kernel implements,
    /// otherwise the topmost tier the CPU reports.
    pub(crate) fn resolve(ladder: &[Tier]) -> Tier {
        if let Some(forced) = Self::requested() {
            if forced.is_available() {
                for &tier in ladder {
                    if tier.rank() <= forced.rank() {
                        return tier;
                    }
                }
            }
        }
        for &tier in ladder {
            if tier.is_available() {
                return tier;
            }
        }
        Tier::Scalar
    }

    #[cfg(test)]
    pub(crate) fn requested() -> Option<Tier> {
        FORCED.with(|cell| cell.get()).or_else(Self::env_tier)
    }

    /// With the opt-in `simd-tier-override` feature the env override also applies to
    /// normal builds (CI and `cargo bench`), which is the only way to measure a
    /// specific path on a machine that has several. The cost is one cached lookup per
    /// matmul call, so the feature is never on by default.
    #[cfg(all(not(test), feature = "simd-tier-override"))]
    pub(crate) fn requested() -> Option<Tier> {
        Self::env_tier()
    }

    #[cfg(not(any(test, feature = "simd-tier-override")))]
    pub(crate) fn requested() -> Option<Tier> {
        None
    }

    /// `RNNOISE_SIMD_TIER`, read once. Set by CI to exercise one path on a runner.
    #[cfg(any(test, feature = "simd-tier-override"))]
    fn env_tier() -> Option<Tier> {
        static ENV: std::sync::OnceLock<Option<Tier>> = std::sync::OnceLock::new();
        *ENV.get_or_init(|| {
            std::env::var("RNNOISE_SIMD_TIER")
                .ok()
                .as_deref()
                .and_then(Tier::parse)
        })
    }

    /// Force `tier` on the current thread for the lifetime of the returned guard.
    #[must_use = "the guard restores the detected ladder when it is dropped"]
    #[cfg(test)]
    pub(crate) fn force(tier: Tier) -> Forced {
        let previous = FORCED.with(|cell| cell.replace(Some(tier)));
        Forced { previous }
    }
}

#[cfg(test)]
thread_local! {
    static FORCED: Cell<Option<Tier>> = const { Cell::new(None) };
}

#[cfg(test)]
pub(crate) struct Forced {
    previous: Option<Tier>,
}

#[cfg(test)]
impl Drop for Forced {
    fn drop(&mut self) {
        FORCED.with(|cell| cell.set(self.previous));
    }
}
