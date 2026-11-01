//! Per-tier CI coverage: run the real pipeline once for every tier the runner actually
//! has, and check that each one produces the same samples as the portable reference.
//!
//! `simd::tests` compares a kernel against `scalar` on a synthetic fixture; this module
//! closes the loop on the *whole* pipeline — front-end, pitch search, FFT, dense GEMV,
//! sparse GEMV and the int8 dot products — so a tier that is bit-exact in isolation but
//! wired into the wrong place still fails.
//!
//! Skips gracefully when `test_data/testing.raw` is absent (rule 4 of AGENTS.md), and
//! prints a coverage table so the CI log states exactly which tiers ran and which were
//! skipped because the runner lacks the ISA.

use crate::denoise::DenoiseState;
use crate::params::FRAME_SIZE;
use crate::simd::Tier;
use crate::weights::RnnModel;
use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
const LADDER: &[Tier] = &[
    Tier::Scalar,
    Tier::Sse2,
    Tier::Sse41,
    Tier::Avx2,
    Tier::Vnni,
];

#[cfg(target_arch = "aarch64")]
const LADDER: &[Tier] = &[Tier::Scalar, Tier::Dotprod];

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const LADDER: &[Tier] = &[Tier::Scalar];

fn available_tiers() -> String {
    LADDER
        .iter()
        .filter(|t| t.is_available())
        .map(|t| t.name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// `test_data/testing.raw` as `i16` samples; `None` when the artifact is not there.
fn clip() -> Option<Vec<i16>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/testing.raw");
    let bytes = std::fs::read(path).ok()?;
    if bytes.is_empty() {
        return None;
    }
    Some(
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect(),
    )
}

/// One pass over the clip with `tier` forced on this thread.
fn run(tier: Tier, quantized: bool, input: &[i16]) -> Vec<f32> {
    let forced = Tier::force(tier);
    let model = if quantized {
        Arc::new(RnnModel::default().quantized())
    } else {
        Arc::new(RnnModel::default())
    };
    let mut st = DenoiseState::new_with_model(model);
    let mut out_buf = [0.0f32; FRAME_SIZE];
    let mut frame = [0.0f32; FRAME_SIZE];
    let mut output = Vec::with_capacity(input.len());
    let mut first = true;
    for chunk in input.as_chunks::<FRAME_SIZE>().0 {
        for (dst, &src) in frame.iter_mut().zip(chunk) {
            *dst = src as f32;
        }
        st.process_frame(&mut out_buf, &frame);
        if !first {
            output.extend_from_slice(&out_buf);
        }
        first = false;
    }
    drop(forced);
    output
}

#[test]
fn every_available_tier_matches_the_reference_pipeline() {
    // A tier asked for through `RNNOISE_SIMD_TIER` is coverage CI promised, so it must
    // actually run: a runner without the ISA fails here rather than printing a skip.
    if let Some(forced) = Tier::requested() {
        assert!(
            forced.is_available(),
            "coverage: RNNOISE_SIMD_TIER={} cannot run on this runner ({} available)",
            forced.name(),
            available_tiers(),
        );
    }
    let Some(input) = clip() else {
        eprintln!("coverage: test_data/testing.raw absent, skipped");
        return;
    };
    let reference = run(Tier::Scalar, false, &input);
    let int8_reference = run(Tier::Scalar, true, &input);
    eprintln!(
        "simd coverage ({} frames, {} samples):",
        input.len() / FRAME_SIZE,
        reference.len()
    );
    for &tier in LADDER {
        if !tier.is_available() {
            eprintln!(
                "  {:>8}  skipped: runner has no {}",
                tier.name(),
                tier.name()
            );
            continue;
        }
        for (path, quantized, baseline) in [
            ("float", false, &reference),
            ("int8", true, &int8_reference),
        ] {
            let out = run(tier, quantized, &input);
            let differing = out.iter().zip(baseline).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing,
                0,
                "{path} path diverged from the scalar reference on tier {}",
                tier.name()
            );
            eprintln!("  {:>8}  {path}: bit-identical to scalar", tier.name());
        }
    }
}
