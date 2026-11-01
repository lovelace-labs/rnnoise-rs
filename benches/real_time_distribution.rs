//! Per-frame **distribution** benchmark — the real-time view of `cargo bench`.
//!
//! `benches/process.rs` (criterion) measures throughput and saves the CI baseline. That
//! answers "how fast on average", which is the wrong question for an audio callback: it is
//! bounded by its worst frame. This bench times each frame individually and reports the
//! distribution (mean, p50, p99, p99.9, max) plus the headroom against the 10 ms frame
//! budget, and it measures cold start separately, because construction is where the one-time
//! work lives (tables, weight parse, the CPUID probe). See `doc/realtime.md`.
//!
//! `Instant` adds ~30–50 ns per sample and is included. The p99.9/max tail is OS noise
//! (scheduler, IRQ, page faults), not algorithmic work: every loop in the frame path is
//! fixed-size, so the work per frame is bounded by the model geometry.
//!
//! Run: `make bench-dist` or `cargo bench --all-features --bench real_time_distribution`.

// Benchmarks are not public API, so they carry no docs.
#![allow(missing_docs)]

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use rnnoise::{warmup, DenoiseState, RnnModel, FRAME_SIZE};

/// One 10 ms frame, in ns — the budget a real-time callback has to fit inside.
const FRAME_BUDGET_NS: f64 = 10_000_000.0;

struct Stats {
    mean: f64,
    p50: f64,
    p99: f64,
    p99_9: f64,
    max: f64,
}

#[must_use]
fn stats(times: &[u64]) -> Stats {
    let mut sorted = times.to_vec();
    sorted.sort_unstable();
    let sum: u128 = sorted.iter().map(|&v| v as u128).sum();
    let at = |q: usize| sorted[sorted.len() * q / 1000] as f64;
    Stats {
        mean: (sum as f64) / sorted.len() as f64,
        p50: at(500),
        p99: at(990),
        p99_9: at(999),
        max: sorted[sorted.len() - 1] as f64,
    }
}

fn report(label: &str, s: &Stats) {
    println!(
        "{label}  mean {:>8.1}  p50 {:>8.1}  p99 {:>8.1}  p99.9 {:>8.1}  max {:>8.1} µs   \
         headroom on the 10 ms budget: mean {:.1}×, max {:.1}×",
        s.mean / 1000.0,
        s.p50 / 1000.0,
        s.p99 / 1000.0,
        s.p99_9 / 1000.0,
        s.max / 1000.0,
        FRAME_BUDGET_NS / s.mean,
        FRAME_BUDGET_NS / s.max,
    );
}

#[must_use]
fn measure(
    st: &mut DenoiseState,
    frame: &[f32; FRAME_SIZE],
    out: &mut [f32; FRAME_SIZE],
    warmup_frames: usize,
    iters: usize,
) -> Stats {
    for _ in 0..warmup_frames {
        st.process_frame(out, frame);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        st.process_frame(black_box(out), black_box(frame));
        times.push(start.elapsed().as_nanos() as u64);
    }
    stats(&times)
}

#[cfg(feature = "legacy-model")]
#[must_use]
fn measure_v1(
    st: &mut rnnoise::DenoiseStateV1,
    frame: &[f32; FRAME_SIZE],
    out: &mut [f32; FRAME_SIZE],
    warmup_frames: usize,
    iters: usize,
) -> Stats {
    for _ in 0..warmup_frames {
        st.process_frame(out, frame);
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        st.process_frame(black_box(out), black_box(frame));
        times.push(start.elapsed().as_nanos() as u64);
    }
    stats(&times)
}

/// Which tier `src/simd` will actually pick on this machine (runtime, not build time).
#[cfg(target_arch = "x86_64")]
#[must_use]
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

#[cfg(not(target_arch = "x86_64"))]
#[must_use]
fn detected_tier() -> &'static str {
    "scalar (or dotprod on aarch64)"
}

fn main() {
    println!(
        "simd tier selected by runtime detection: {}",
        detected_tier()
    );
    let mut seed = 1u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0 - 0.5
    };
    // Active frame: tone + noise (never silent -> the network always runs, i.e. the worst
    // case per frame).
    let mut active = [0.0f32; FRAME_SIZE];
    for (i, s) in active.iter_mut().enumerate() {
        *s = 3000.0 * (i as f32 * 0.05).sin() + 2000.0 * rng();
    }
    // Silent frame: NN is skipped (front-end only).
    let silent = [0.0f32; FRAME_SIZE];

    let iters = 20_000;
    let mut out = [0.0f32; FRAME_SIZE];

    println!("\ncold start (fresh process, before any frame):");
    let start = Instant::now();
    let mut st = DenoiseState::new();
    let construct = start.elapsed().as_nanos() as f64;
    let start = Instant::now();
    st.process_frame(&mut out, &active);
    let first_frame = start.elapsed().as_nanos() as f64;
    println!(
        "  construction {:>8.1} µs   first frame {:>8.1} µs   (warmup() resolves CPUID here, so \
         no first-use probe is left inside the frame path)",
        construct / 1000.0,
        first_frame / 1000.0
    );

    println!("\nrnnoise-rs (float, default) per-frame ({iters} iters, warmup before the window):");
    let full = measure(&mut st, &active, &mut out, 200, iters);
    report("  full (active)   ", &full);
    let frontend = measure(&mut st, &silent, &mut out, 200, iters);
    report("  front-end (sil)  ", &frontend);
    println!(
        "  -> neural net ≈   mean {:>8.1} µs   ({:.0}% of full)",
        (full.mean - frontend.mean) / 1000.0,
        100.0 * (full.mean - frontend.mean) / full.mean
    );

    println!("\nrnnoise-rs (int8 quantized) per-frame:");
    let q = Arc::new(RnnModel::default().quantized());
    let mut qst = DenoiseState::new_with_model(q);
    let quant_full = measure(&mut qst, &active, &mut out, 200, iters);
    report("  full (active)   ", &quant_full);
    let quant_front = measure(&mut qst, &silent, &mut out, 200, iters);
    report("  front-end (sil)  ", &quant_front);
    println!(
        "  -> neural net ≈   mean {:>8.1} µs   (vs {:.1} float → {:.2}× speedup)",
        (quant_full.mean - quant_front.mean) / 1000.0,
        (full.mean - frontend.mean) / 1000.0,
        (full.mean - frontend.mean) / (quant_full.mean - quant_front.mean)
    );

    #[cfg(feature = "legacy-model")]
    {
        use rnnoise::DenoiseStateV1;
        warmup();
        println!("\nrnnoise-rs (legacy / old model) per-frame:");
        let mut lst = DenoiseStateV1::new();
        let legacy_full = measure_v1(&mut lst, &active, &mut out, 200, iters);
        report("  full (active)   ", &legacy_full);
        let legacy_front = measure_v1(&mut lst, &silent, &mut out, 200, iters);
        report("  front-end (sil)  ", &legacy_front);
        println!(
            "  -> neural net ≈   mean {:>8.1} µs",
            (legacy_full.mean - legacy_front.mean) / 1000.0
        );
    }
}
