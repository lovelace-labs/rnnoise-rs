//! Per-frame throughput micro-benchmark (no external dependencies).
//!
//! Reports the steady-state cost of `process_frame` on an active (never-silent)
//! frame, plus an ablation: feeding silence skips the neural network, so
//! `full - frontend ≈ NN cost`. Useful for the speed-up analysis in
//! `BENCHMARKS.md`.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use rnnoise::{DenoiseState, RnnModel, FRAME_SIZE};

fn bench(label: &str, frame: &[f32; FRAME_SIZE], iters: usize) -> f64 {
    bench_with(label, DenoiseState::new(), frame, iters)
}

fn bench_with(label: &str, mut st: DenoiseState, frame: &[f32; FRAME_SIZE], iters: usize) -> f64 {
    let mut out = [0.0f32; FRAME_SIZE];
    for _ in 0..200 {
        st.process_frame(&mut out, frame);
    }
    let start = Instant::now();
    for _ in 0..iters {
        st.process_frame(black_box(&mut out), black_box(frame));
    }
    let ns = start.elapsed().as_nanos() as f64 / iters as f64;
    let rtf = (FRAME_SIZE as f64 / 48_000.0) / (ns / 1e9);
    println!(
        "{label:<22} {:>9.1} µs/frame   {:>6.1}× real time",
        ns / 1000.0,
        rtf
    );
    ns
}

fn main() {
    let mut seed = 1u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0 - 0.5
    };
    // Active frame: tone + noise (never silent -> NN always runs).
    let mut active = [0.0f32; FRAME_SIZE];
    for (i, s) in active.iter_mut().enumerate() {
        *s = 3000.0 * (i as f32 * 0.05).sin() + 2000.0 * rng();
    }
    // Silent frame: NN is skipped (front-end only).
    let silent = [0.0f32; FRAME_SIZE];

    let iters = 20_000;
    println!("rnnoise-rs (float, default) per-frame ({iters} iters):");
    let full = bench("  full (active)", &active, iters);
    let frontend = bench("  front-end (silent)", &silent, iters);
    let nn = full - frontend;
    println!(
        "  -> neural net ≈    {:>9.1} µs/frame   ({:.0}% of full)",
        nn / 1000.0,
        100.0 * nn / full
    );

    println!("\nrnnoise-rs (int8 quantized) per-frame:");
    let q = Arc::new(RnnModel::default().quantized());
    let qfull = bench_with(
        "  full (active)",
        DenoiseState::with_model(q.clone()),
        &active,
        iters,
    );
    let qfront = bench_with(
        "  front-end (silent)",
        DenoiseState::with_model(q),
        &silent,
        iters,
    );
    println!(
        "  -> neural net ≈    {:>9.1} µs/frame   (vs {:.1} float → {:.2}× speedup)",
        (qfull - qfront) / 1000.0,
        nn / 1000.0,
        nn / (qfull - qfront)
    );
    println!(
        "  -> full frame      {:>9.1} µs/frame   ({:.1}× real time, {:.2}× vs float full)",
        qfull / 1000.0,
        (FRAME_SIZE as f64 / 48_000.0) / (qfull / 1e9),
        full / qfull
    );
}
