//! FFT tests: the plan against a naive DFT, and every vector tier against the
//! portable reference. A tier whose CPU feature is not present is skipped.

use crate::fft::{plan::KissFft, scalar, spectrum::Cpx};

fn seed_input(seed: u32, n: usize) -> Vec<Cpx> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f32 / 16_777_216.0 - 0.5
    };
    (0..n).map(|_| Cpx::new(next(), 0.0)).collect()
}

fn naive_dft(input: &[Cpx], scale: f32) -> Vec<Cpx> {
    let n = input.len();
    let mut out = vec![Cpx::default(); n];
    for (k, o) in out.iter_mut().enumerate() {
        let mut acc = Cpx::default();
        for (j, &x) in input.iter().enumerate() {
            let phase = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
            let (s, c) = phase.sin_cos();
            let (c, s) = (c as f32, s as f32);
            acc.r += x.r * c - x.i * s;
            acc.i += x.r * s + x.i * c;
        }
        *o = Cpx {
            r: acc.r * scale,
            i: acc.i * scale,
        };
    }
    out
}

#[test]
fn factors_960() {
    let fft = KissFft::new(960);
    // 960 = 5 * 3 * 4 * 4 * 4, in the order `rnn_fft_impl` walks the stages.
    assert_eq!(&fft.factors[0..10], &[5, 192, 3, 64, 4, 16, 4, 4, 4, 1]);
    let stage_order: Vec<(usize, usize)> = fft.stages.iter().map(|s| (s.p, s.m)).collect();
    assert_eq!(
        stage_order,
        vec![(4, 1), (4, 4), (4, 16), (3, 64), (5, 192)],
        "stage order must match kiss_fft's fast degenerate case"
    );
}

#[test]
fn fft_matches_naive_dft_960() {
    let n = 960;
    let fft = KissFft::new(n);
    let input = seed_input(12345, n);
    let mut out = vec![Cpx::default(); n];
    fft.forward(&input, &mut out);
    let reference = naive_dft(&input, 1.0 / n as f32);
    let mut max_err = 0.0f32;
    for (a, b) in out.iter().zip(&reference) {
        max_err = max_err.max((a.r - b.r).abs()).max((a.i - b.i).abs());
    }
    assert!(max_err < 1e-5, "max FFT error {max_err}");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn vector_tier_matches_scalar_bit_for_bit() {
    // Sizes with different factorisations, so tiers and tails are both exercised.
    for n in [960usize, 480, 320, 256, 64] {
        let fft = KissFft::new(n);
        let input = seed_input(999, n);
        let mut reference = vec![Cpx::default(); n];
        let mut tier = vec![Cpx::default(); n];
        for i in 0..n {
            let dst = fft.bitrev[i] as usize;
            reference[dst] = Cpx::new(fft.scale * input[i].r, fft.scale * input[i].i);
            tier[dst] = reference[dst];
        }
        for (si, st) in fft.stages.iter().enumerate() {
            scalar::stage(st, &mut reference);
            unsafe { crate::fft::x86_64::process(&fft.stages[si..si + 1], &mut tier) };
            let diffs: Vec<usize> = (0..n).filter(|i| tier[*i] != reference[*i]).collect();
            if !diffs.is_empty() {
                println!(
                    "nfft {n} stage {si} (p={}, m={}, n={}, mm={}): {} diffs, first idx {}",
                    st.p,
                    st.m,
                    st.n,
                    st.mm,
                    diffs.len(),
                    diffs[0]
                );
                let d = diffs[0];
                println!("  tier={:?} ref={:?}", tier[d], reference[d]);
            }
        }
        assert_eq!(tier, reference, "SSE2 tier diverged for nfft {n}");
    }
}

// `cargo test --release -- --ignored --nocapture`
#[test]
#[ignore]
fn fft_timing() {
    use std::hint::black_box;
    use std::time::Instant;
    for n in [960usize, 480] {
        let fft = KissFft::new(n);
        let inp = vec![Cpx::new(0.3, -0.1); n];
        let mut out = vec![Cpx::default(); n];
        for _ in 0..2000 {
            fft.forward(&inp, &mut out);
        }
        let iters = 200_000;
        let t = Instant::now();
        for _ in 0..iters {
            fft.forward(black_box(&inp), black_box(&mut out));
        }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        println!("complex FFT n={n}: {:.3} µs", ns / 1000.0);
    }
}
