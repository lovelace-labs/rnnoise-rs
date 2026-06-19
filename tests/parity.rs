//! End-to-end parity against the upstream C reference (`rnnoise_demo`, scalar
//! build) on `test_data/testing.raw`. The reference output was produced with
//! the same float model embedded here.

use rnnoise::{DenoiseState, FRAME_SIZE};

fn load_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn matches_c_reference() {
    let input = load_i16(include_bytes!("../test_data/testing.raw"));
    let reference = load_i16(include_bytes!("../test_data/ref_out.raw"));

    let mut st = DenoiseState::new();
    let mut out_buf = [0.0f32; FRAME_SIZE];
    let mut output: Vec<f32> = Vec::new();
    let mut first = true;
    for chunk in input.chunks_exact(FRAME_SIZE) {
        let frame: Vec<f32> = chunk.iter().map(|&s| s as f32).collect();
        st.process_frame(&mut out_buf, &frame);
        if !first {
            output.extend_from_slice(&out_buf);
        }
        first = false;
    }

    assert_eq!(output.len(), reference.len(), "output length mismatch");

    // Convert like the C demo (float -> short truncates toward zero).
    let out_i16: Vec<i16> = output.iter().map(|&x| x as i16).collect();

    let mut max_diff = 0i32;
    let mut nonzero = 0usize;
    let mut ss_ref = 0.0f64;
    let mut ss_diff = 0.0f64;
    for (&r, &o) in reference.iter().zip(&out_i16) {
        let d = (r as i32 - o as i32).abs();
        if d != 0 {
            nonzero += 1;
        }
        max_diff = max_diff.max(d);
        ss_ref += (r as f64).powi(2);
        ss_diff += (d as f64).powi(2);
    }
    let rel = ss_diff / ss_ref;
    eprintln!(
        "parity: samples={} nonzero_diff={} max_diff={} rel_energy={:.3e}",
        reference.len(),
        nonzero,
        max_diff,
        rel
    );

    // Far tighter than nnnoiseless's 1e-4 acceptance; differences are limited to
    // a handful of 1-LSB roundings vs the C build (see TODO.md parity notes).
    assert!(rel < 1e-6, "relative energy diff too large: {rel:.3e}");
    assert!(max_diff <= 2, "max sample diff too large: {max_diff}");
}
