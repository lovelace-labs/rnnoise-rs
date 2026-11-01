//! Per-frame throughput benchmark for `cargo bench`.
//!
//! Criterion groups:
//!
//! - `float`     — the default (float model) path;
//! - `int8`      — `RnnModel::quantized()`, the opt-in int8 path;
//! - `legacy`    — the optional old model (`legacy-model` feature);
//! - each group has an `active` (never silent, so the network runs) and a `silent`
//!   (front-end only, network skipped) input, which is the ablation `BENCHMARKS.md`
//!   uses to split front-end cost from neural-net cost.
//!
//! The header prints the SIMD tier runtime detection selected, so a number from a CI
//! runner can be attributed to a tier. `RNNOISE_SIMD_TIER` applies only when the crate
//! is built with the opt-in `simd-tier-override` feature (CI's bench job sets both), so
//! one path can be measured against another on the same machine.
//!
//! CI runs `cargo bench -- --save-baseline ci` and uploads the JSON as an artifact.

// `criterion_group!` generates items that `missing_docs` cannot be satisfied for.
#![allow(missing_docs)]

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use rnnoise::{DenoiseState, RnnModel, FRAME_SIZE};

/// Which tier `src/simd` will actually pick on this machine (runtime, not build time).
#[cfg(target_arch = "x86_64")]
fn detected_tier() -> &'static str {
    for (name, on) in [
        (
            "v4 avx512f+avx512bw+avx512vnni",
            std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni"),
        ),
        ("v3 avx2", std::arch::is_x86_feature_detected!("avx2")),
        ("v2 sse4.1", std::arch::is_x86_feature_detected!("sse4.1")),
        ("v1 sse2", std::arch::is_x86_feature_detected!("sse2")),
    ] {
        if on {
            return name;
        }
    }
    "scalar"
}

#[cfg(target_arch = "aarch64")]
fn detected_tier() -> &'static str {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        "dotprod"
    } else {
        "scalar"
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detected_tier() -> &'static str {
    "scalar"
}

/// Tone + noise: never silent, so the neural network runs on every frame.
fn active_frame() -> [f32; FRAME_SIZE] {
    let mut seed = 1u32;
    let mut frame = [0.0f32; FRAME_SIZE];
    for (i, s) in frame.iter_mut().enumerate() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = (seed >> 8) as f32 / 16_777_216.0 - 0.5;
        *s = 3000.0 * (i as f32 * 0.05).sin() + 2000.0 * noise;
    }
    frame
}

fn measure(
    criterion: &mut Criterion,
    group: &str,
    state: &mut DenoiseState,
    active: &[f32; FRAME_SIZE],
) {
    let mut out = [0.0f32; FRAME_SIZE];
    // Warm up: build the shared tables and run a few frames before measuring.
    for _ in 0..200 {
        state.process_frame(&mut out, active);
    }
    let silent = [0.0f32; FRAME_SIZE];
    let mut g = criterion.benchmark_group(group);
    // One frame is 10 ms of audio at 48 kHz, so the "real time" factor in the summary
    // comes straight from these numbers: 10 ms / sample_time.
    g.throughput(Throughput::Elements(1));
    g.bench_function("active", |b| {
        b.iter(|| state.process_frame(black_box(&mut out), black_box(active)))
    });
    g.bench_function("silent", |b| {
        b.iter(|| state.process_frame(black_box(&mut out), black_box(&silent)))
    });
    g.finish();
}

fn bench(criterion: &mut Criterion) {
    eprintln!("simd tier detected: {}", detected_tier());
    match std::env::var("RNNOISE_SIMD_TIER") {
        Ok(requested) if !requested.is_empty() => eprintln!(
            "simd tier requested: {requested} (honoured because `simd-tier-override` is enabled)"
        ),
        _ => {}
    }

    let active = active_frame();

    let mut float_state = DenoiseState::new();
    measure(criterion, "float", &mut float_state, &active);

    let quantized = Arc::new(RnnModel::default().quantized());
    let mut int8_state = DenoiseState::new_with_model(quantized);
    measure(criterion, "int8", &mut int8_state, &active);

    #[cfg(feature = "legacy-model")]
    {
        use rnnoise::DenoiseStateV1;
        let mut legacy = DenoiseStateV1::new();
        let mut out = [0.0f32; FRAME_SIZE];
        for _ in 0..200 {
            legacy.process_frame(&mut out, &active);
        }
        let silent = [0.0f32; FRAME_SIZE];
        let mut g = criterion.benchmark_group("legacy");
        g.bench_function("active", |b| {
            b.iter(|| legacy.process_frame(black_box(&mut out), black_box(&active)))
        });
        g.bench_function("silent", |b| {
            b.iter(|| legacy.process_frame(black_box(&mut out), black_box(&silent)))
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
