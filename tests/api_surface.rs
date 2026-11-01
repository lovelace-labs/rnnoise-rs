//! Public-API surface gate.
//!
//! Every item the crate exports is named here, in the shape the docs promise. If a
//! public item is renamed, removed, or its signature changes, this file stops
//! compiling — which turns an API change into a build failure instead of a silent
//! one.
//!
//! For the cross-crate view run `cargo public-api diff <last-release>`.

use std::sync::Arc;

use rnnoise::ModelError;
use rnnoise::{DenoiseState, RnnModel, FRAME_SIZE, FREQ_SIZE, NB_BANDS, NB_FEATURES, WINDOW_SIZE};

#[test]
fn crate_root_surface_matches_the_docs() {
    assert_eq!(FRAME_SIZE, 480, "frame geometry is part of the public API");
    assert_eq!(
        WINDOW_SIZE,
        2 * FRAME_SIZE,
        "50 % overlap is part of the public API"
    );
    assert_eq!(
        FREQ_SIZE,
        FRAME_SIZE + 1,
        "bin count is part of the public API"
    );
    assert_eq!(NB_BANDS, 32, "band count is part of the public API");
    assert_eq!(
        NB_FEATURES,
        2 * NB_BANDS + 1,
        "feature vector length is part of the public API"
    );
}

#[test]
fn documented_constructors_exist() {
    // `new()` and `new_with_model()` (constructors are `new` / `new_with_foo`;
    // `with_foo` is the closure-accessor form).
    let model = RnnModel::default();
    let shared = Arc::new(model.clone());
    let mut st = DenoiseState::new();
    let mut shared_st = DenoiseState::new_with_model(shared);
    let mut out = [0.0f32; FRAME_SIZE];
    let input = [0.0f32; FRAME_SIZE];
    let vad = st.process_frame(&mut out, &input);
    assert!(
        (0.0..=1.0).contains(&vad),
        "process_frame returns a VAD probability"
    );
    let _ = shared_st.process_frame(&mut out, &input);
}

#[test]
fn error_type_is_public_and_open() {
    // `#[non_exhaustive]` means callers may match the known variants but must keep a
    // wildcard arm, so adding a variant later is not a breaking change.
    let err = RnnModel::from_bytes(&[0u8; 4]).expect_err("a truncated blob must not load");
    let label = match err {
        ModelError::Malformed => "malformed".to_string(),
        ModelError::MissingArray(n) => format!("missing array {n}"),
        ModelError::BadSize(n) => format!("bad size {n}"),
        // The wildcard arm is what `#[non_exhaustive]` forces on callers, which is the
        // point: adding a variant later must not break this file.
        other => format!("unhandled variant {other:?}"),
    };
    assert!(!label.is_empty(), "the error type must be printable");
}

#[cfg(feature = "legacy-model")]
#[test]
fn legacy_surface_matches_the_docs() {
    use rnnoise::DenoiseStateV1;
    use rnnoise::RnnModelV1;

    let model = Arc::new(RnnModelV1::default());
    let mut st = DenoiseStateV1::new_with_model(model);
    let mut out = [0.0f32; FRAME_SIZE];
    let input = [0.0f32; FRAME_SIZE];
    let vad = st.process_frame(&mut out, &input);
    assert!(
        (0.0..=1.0).contains(&vad),
        "the legacy path returns the same kind of value"
    );
}
