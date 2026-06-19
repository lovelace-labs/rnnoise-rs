//! Minimal throughput benchmark (no external dependencies). Reports
//! frames/second and the real-time factor for `process_frame`.

use std::hint::black_box;
use std::time::Instant;

use rnnoise::{DenoiseState, FRAME_SIZE};

fn main() {
    let mut st = DenoiseState::new();

    // A deterministic noisy signal: tone + pseudo-random noise.
    let mut seed = 1u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0 - 0.5
    };
    let frame: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| 3000.0 * (i as f32 * 0.05).sin() + 2000.0 * rng())
        .collect();
    let mut out = vec![0.0f32; FRAME_SIZE];

    // Warm up.
    for _ in 0..100 {
        st.process_frame(&mut out, &frame);
    }

    let iters: usize = 20_000;
    let start = Instant::now();
    for _ in 0..iters {
        st.process_frame(black_box(&mut out), black_box(&frame));
    }
    let elapsed = start.elapsed();

    let per_frame = elapsed / iters as u32;
    let audio_secs = (iters * FRAME_SIZE) as f64 / 48_000.0;
    let rtf = audio_secs / elapsed.as_secs_f64();
    println!("frames:        {iters}");
    println!("total:         {elapsed:?}");
    println!("per frame:     {per_frame:?}");
    println!("audio/wall:    {rtf:.1}x real time");
}
