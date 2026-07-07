//! Standalone throughput benchmark for the legacy model, isolated from the
//! other benches so the core isn't thermally throttled by prior work.
//! `cargo run --release --features legacy-model --example bench_legacy`

#[cfg(feature = "legacy-model")]
fn main() {
    use rnnoise::{DenoiseStateV1, FRAME_SIZE};
    use std::hint::black_box;
    use std::time::Instant;

    let mut seed = 1u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0 - 0.5
    };
    let mut frame = [0.0f32; FRAME_SIZE];
    for (i, s) in frame.iter_mut().enumerate() {
        *s = 3000.0 * (i as f32 * 0.05).sin() + 2000.0 * rng();
    }

    let mut st = DenoiseStateV1::new();
    let mut out = [0.0f32; FRAME_SIZE];
    for _ in 0..500 {
        st.process_frame(&mut out, &frame);
    }
    let iters = 50_000;
    let t = Instant::now();
    for _ in 0..iters {
        st.process_frame(black_box(&mut out), black_box(&frame));
    }
    let ns = t.elapsed().as_nanos() as f64 / iters as f64;
    println!(
        "full (active) {:.2} µs/frame   {:.0}× real time",
        ns / 1000.0,
        (FRAME_SIZE as f64 / 48_000.0) / (ns / 1e9)
    );
}

#[cfg(not(feature = "legacy-model"))]
fn main() {
    eprintln!("run with --features legacy-model");
}
