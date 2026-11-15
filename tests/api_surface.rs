//! Public-API surface gate.
//!
//! Every item the crate exports is named here, in the shape the docs promise. If a
//! public item is renamed, removed, or its signature changes, this file stops
//! compiling — which turns an API change into a build failure instead of a silent
//! one.
//!
//! For the cross-crate view run `cargo public-api diff <last-release>`.

#[cfg(any(feature = "embedded-model", feature = "embedded-model-int8"))]
use std::sync::Arc;

#[cfg(any(feature = "embedded-model", feature = "embedded-model-int8"))]
use rnnoise::warmup;
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

#[cfg(any(feature = "embedded-model", feature = "embedded-model-int8"))]
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

#[cfg(any(feature = "embedded-model", feature = "embedded-model-int8"))]
#[test]
fn real_time_surface_is_named() {
    // `warmup()` is part of the public contract: it resolves the CPUID cache that
    // the tier ladder uses, so the first frame is not the first probe. It must be callable
    // before any frame and must not change results.
    warmup();
    let mut st = DenoiseState::new();
    let mut out = [0.0f32; FRAME_SIZE];
    let input = [0.0f32; FRAME_SIZE];
    let vad = st.process_frame(&mut out, &input);
    assert!(
        (0.0..=1.0).contains(&vad),
        "warmup must not change the frame path"
    );
}

#[test]
fn try_new_compiles_in_every_feature_configuration() {
    // `try_new()` is the constructor that exists with or without an embedded model: it either
    // constructs, or it reports that there is no default model to use. That is the whole point of
    // making embedding optional — a build with no blob has no model, and must say so.
    match DenoiseState::try_new() {
        Ok(mut st) => {
            let mut out = [0.0f32; FRAME_SIZE];
            let vad = st.process_frame(&mut out, &[0.0f32; FRAME_SIZE]);
            assert!((0.0..=1.0).contains(&vad));
        }
        Err(ModelError::NoEmbeddedModel) => {}
        Err(other) => assert!(
            matches!(other, ModelError::NoEmbeddedModel),
            "unexpected error from try_new: {other}"
        ),
    }
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

#[cfg(feature = "model-loader")]
#[test]
fn loader_surface_matches_the_docs() {
    // The on-demand path: `from_file` and `from_file_with_checksum`, plus the public `crc32` a
    // caller uses to pin a downloaded blob. Nothing here reads a real artifact — the point is that
    // the documented names and signatures exist.
    let from_file: fn(&std::path::Path) -> Result<RnnModel, ModelError> = RnnModel::from_file;
    let pinned: fn(&std::path::Path, u32) -> Result<RnnModel, ModelError> =
        RnnModel::from_file_with_checksum;
    assert!(matches!(
        from_file(std::path::Path::new("models/nope.bin")),
        Err(ModelError::Io(_))
    ));
    assert!(matches!(
        pinned(std::path::Path::new("models/nope.bin"), 0),
        Err(ModelError::Io(_))
    ));
    assert_ne!(
        rnnoise::crc32(b"DNNw"),
        0,
        "crc32 must be callable from outside"
    );
}

#[cfg(any(feature = "embedded-model", feature = "embedded-model-int8"))]
#[test]
fn the_writer_round_trips_through_the_reader() {
    // `to_bytes` is the mirror of `from_bytes`: whatever this crate writes, it must read back
    // unchanged — including re-verifying the container header it wrote.
    let model = RnnModel::default();
    assert_eq!(RnnModel::from_bytes(&model.to_bytes()), Ok(model.clone()));
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
