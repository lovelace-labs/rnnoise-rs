//! Parity of the optional legacy (old-model) pipeline against the same model as
//! implemented by nnnoiseless (`test_data/legacy_ref.raw`). Requires the
//! `legacy-model` feature.
#![cfg(feature = "legacy-model")]

use rnnoise::{DenoiseStateV1, FRAME_SIZE};

fn load_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn legacy_matches_old_model() {
    let input = load_i16(include_bytes!("../test_data/testing.raw"));
    let reference = load_i16(include_bytes!("../test_data/legacy_ref.raw"));

    let mut st = DenoiseStateV1::new();
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
    assert_eq!(output.len(), reference.len());

    let out_i16: Vec<i16> = output.iter().map(|&x| x as i16).collect();
    let mut ss_ref = 0.0f64;
    let mut ss_diff = 0.0f64;
    let mut max_diff = 0i32;
    let mut nonzero = 0usize;
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
    eprintln!("legacy parity: nonzero={nonzero} max_diff={max_diff} rel_energy={rel:.3e}");
    // KISS-FFT vs nnnoiseless's rustfft accounts for ~1 LSB on some samples,
    // the same spread nnnoiseless itself has vs the original C reference.
    assert!(rel < 1e-4, "legacy deviation too large: {rel:.3e}");
}
