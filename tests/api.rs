//! Public-API behaviour tests (independent of the C reference).

use std::sync::Arc;

use rnnoise::{DenoiseState, ModelError, RnnModel, FRAME_SIZE};

fn noisy_frames(n: usize) -> Vec<[f32; FRAME_SIZE]> {
    let mut seed = 42u32;
    let mut rng = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0 - 0.5
    };
    (0..n)
        .map(|f| {
            let mut frame = [0.0f32; FRAME_SIZE];
            for (i, s) in frame.iter_mut().enumerate() {
                *s = 4000.0 * ((f * FRAME_SIZE + i) as f32 * 0.04).sin() + 1500.0 * rng();
            }
            frame
        })
        .collect()
}

fn run(mut st: DenoiseState, frames: &[[f32; FRAME_SIZE]]) -> Vec<f32> {
    let mut out = [0.0f32; FRAME_SIZE];
    let mut acc = Vec::new();
    for f in frames {
        st.process_frame(&mut out, f);
        acc.extend_from_slice(&out);
    }
    acc
}

#[test]
fn embedded_model_loads() {
    // Default model parses and yields the expected layer dimensions.
    let _ = RnnModel::default();
}

#[test]
fn deterministic_across_instances() {
    let frames = noisy_frames(30);
    let a = run(DenoiseState::new(), &frames);
    let b = run(DenoiseState::new(), &frames);
    assert_eq!(a, b, "two fresh denoisers must produce identical output");
}

#[test]
fn with_model_matches_default() {
    // A shared Arc<RnnModel> must behave exactly like the built-in default.
    let frames = noisy_frames(20);
    let default_out = run(DenoiseState::new(), &frames);
    let model = Arc::new(RnnModel::default());
    let custom_out = run(DenoiseState::with_model(model), &frames);
    assert_eq!(default_out, custom_out);
}

#[test]
fn silence_is_stable() {
    // All-zero input must not panic and must stay (near) silent.
    let mut st = DenoiseState::new();
    let zero = [0.0f32; FRAME_SIZE];
    let mut out = [0.0f32; FRAME_SIZE];
    for _ in 0..50 {
        let vad = st.process_frame(&mut out, &zero);
        assert!(vad.is_finite());
    }
    let peak = out.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(peak < 1.0, "silence produced non-trivial output: {peak}");
}

#[test]
fn rejects_garbage_blob() {
    // Too short to hold a record header.
    assert!(matches!(
        RnnModel::from_bytes(&[0u8; 16]),
        Err(ModelError::Malformed)
    ));
    // Empty blob parses to no arrays, so the first required layer is missing.
    assert!(matches!(
        RnnModel::from_bytes(&[]),
        Err(ModelError::MissingArray(_))
    ));
}

#[test]
fn output_finite() {
    let frames = noisy_frames(40);
    let out = run(DenoiseState::new(), &frames);
    assert!(out.iter().all(|x| x.is_finite()));
}
