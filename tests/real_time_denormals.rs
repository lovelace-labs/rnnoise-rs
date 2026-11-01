//! Real-time gate: the frame path must not hit a data-dependent
//! denormal slow path for realistic input.
//!
//! A `f32` value can only be subnormal below the smallest normal, `1.1754944e-38`. In
//! this pipeline subnormals can only be *produced* by multiplication or squaring (the
//! FFT butterflies, band energy/correlation, the high-pass biquad), so the boundary is an
//! input amplitude of `sqrt(1.1754944e-38) ≈ 1.084e-19` — 19 orders of magnitude below the
//! LSB of 16-bit PCM, i.e. not audio, and not reachable at the amplitudes this crate
//! documents (`i16` range, `-32768.0..=32767.0`). The guards that would otherwise turn a
//! zero into a division (`1e-2 + ex[i]`, `1e-3 + …`, `1e-8 + ep[i]`, `0.001 + …`) stop a
//! subnormal value from becoming a subnormal dividend.
//!
//! So the measurable half of the claim is what this test asserts: over the real test
//! signal, no output sample is subnormal, and the signal's smallest non-zero sample is far
//! above the boundary. FTZ/DAZ is deliberately **not** set by the crate — it would change
//! results for exactly those pathological inputs, and bit-exactness is the contract
//! (`doc/parity.md`). A caller that wants flushing sets it itself; see `doc/realtime.md`.

use rnnoise::{DenoiseState, FRAME_SIZE};

#[must_use]
fn is_subnormal(x: f32) -> bool {
    x != 0.0 && x.to_bits() < (1 << 23)
}

/// Smallest normal `f32`; the subnormal boundary for a product or a square.
const SMALLEST_NORMAL_F32: f32 = 1.175_494_4e-38;

#[test]
fn no_denormals_on_the_test_signal() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/testing.raw");
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("skipping: {path} not present");
        return;
    };

    let samples = bytes.as_chunks::<2>().0;
    if samples.len() < 2 * FRAME_SIZE {
        eprintln!("skipping: {path} too short");
        return;
    }

    let mut st = DenoiseState::new();
    let mut out = [0.0f32; FRAME_SIZE];
    let mut scanned = 0usize;
    let mut subnormal = 0usize;
    let mut min_nonzero = f32::MAX;
    let mut first = true;

    for frame in samples.as_chunks::<FRAME_SIZE>().0 {
        let mut input = [0.0f32; FRAME_SIZE];
        for (i, pair) in frame.iter().enumerate() {
            let s = i16::from_le_bytes([pair[0], pair[1]]) as f32;
            if s != 0.0 {
                min_nonzero = min_nonzero.min(s.abs());
            }
            input[i] = s;
        }
        st.process_frame(&mut out, &input);
        if first {
            first = false;
            continue; // the one-frame algorithmic delay: frame 1 outputs nothing yet
        }
        for &v in &out {
            scanned += 1;
            if is_subnormal(v) {
                subnormal += 1;
            }
        }
    }

    let boundary = SMALLEST_NORMAL_F32.sqrt();
    eprintln!(
        "denormals: samples={scanned} subnormal={subnormal} \n\
         smallest non-zero input sample {min_nonzero} — {:.0} orders of magnitude above the \n\
         product/square boundary {boundary:.3e}, so no operation here can produce a subnormal",
        min_nonzero.log10() - boundary.log10()
    );

    assert!(scanned > 0, "no samples processed");
    assert_eq!(
        subnormal, 0,
        "the frame path produced subnormal output samples"
    );
    assert!(
        min_nonzero > boundary,
        "input amplitudes are below the denormal boundary: {min_nonzero}"
    );
}
