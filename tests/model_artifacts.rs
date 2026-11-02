//! The committed model artifacts: provenance, pinned checksums, and the errors a bad blob must
//! produce.
//!
//! These tests read the artifacts at runtime rather than with `include_bytes!`, so a checkout
//! without the multi-megabyte blobs skips instead of failing (AGENTS rule 4).

#[cfg(feature = "model-loader")]
use std::path::Path;

use rnnoise::{crc32, DenoiseState, ModelError, RnnModel, FRAME_SIZE};

/// The record stream of the committed float blob, as pinned in `doc/models.md`.
const FLOAT_BLOB_CRC: u32 = 0x1916_93b7;

/// The record stream of the committed int8 blob (`RNNS` payload, so header-independent).
const INT8_BLOB_CRC: u32 = 0x4e5c_4930;

const FLOAT_BLOB: &str = "models/rnnoise_default.bin";
const INT8_BLOB: &str = "models/rnnoise_default_int8.bin";

fn read_or_skip(path: &str) -> Option<Vec<u8>> {
    match std::fs::read(path) {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            eprintln!("skipping: {path} not present ({e})");
            None
        }
    }
}

/// A deterministic noisy frame set — the same shape as `tests/api.rs`.
fn noisy_frames(n: usize) -> Vec<[f32; FRAME_SIZE]> {
    let mut seed = 42u32;
    let mut rng = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16_777_216.0 - 0.5
    };
    (0..n)
        .map(|_| {
            let mut frame = [0.0f32; FRAME_SIZE];
            for (i, s) in frame.iter_mut().enumerate() {
                *s = 4000.0 * ((i as f32) * 0.04).sin() + 1500.0 * rng();
            }
            frame
        })
        .collect()
}

fn run(model: &RnnModel, frames: &[[f32; FRAME_SIZE]]) -> Vec<f32> {
    let mut st = DenoiseState::new_with_model(std::sync::Arc::new(model.clone()));
    let mut out = [0.0f32; FRAME_SIZE];
    let mut acc = Vec::new();
    for frame in frames {
        st.process_frame(&mut out, frame);
        acc.extend_from_slice(&out);
    }
    acc
}

#[test]
fn pinned_checksums_match_the_committed_artifacts() {
    let float = read_or_skip(FLOAT_BLOB);
    let int8 = read_or_skip(INT8_BLOB);
    if let Some(bytes) = &float {
        // A bare stream: the pinned value is the CRC32 of the whole file.
        assert_eq!(crc32(bytes), FLOAT_BLOB_CRC, "{FLOAT_BLOB} changed");
    }
    if let Some(bytes) = &int8 {
        // A container: the pinned value is the CRC32 of the record stream only, so the header
        // itself can change without invalidating it.
        assert_eq!(crc32(&bytes[16..]), INT8_BLOB_CRC, "{INT8_BLOB} changed");
        assert_eq!(
            &bytes[..4],
            b"RNNS",
            "the int8 artifact is written as a container"
        );
    }
}

#[test]
fn the_int8_artifact_is_exactly_what_the_writer_produces() {
    let Some(float) = read_or_skip(FLOAT_BLOB) else {
        return;
    };
    let Some(int8) = read_or_skip(INT8_BLOB) else {
        return;
    };
    // No separate generator algorithm: the committed artifact must be byte-for-byte what
    // `from_bytes(float).quantized().to_bytes()` produces. If this fails, the artifact and the
    // library disagree about what "the int8 model" is.
    let expected = RnnModel::from_bytes(&float)
        .expect("the committed float blob must load")
        .quantized()
        .to_bytes();
    assert_eq!(
        expected, int8,
        "{INT8_BLOB} is not the crate's own quantized model"
    );
}

#[test]
fn the_int8_artifact_behaves_like_the_runtime_quantization() {
    let Some(float) = read_or_skip(FLOAT_BLOB) else {
        return;
    };
    let Some(int8) = read_or_skip(INT8_BLOB) else {
        return;
    };
    let frames = noisy_frames(20);
    let runtime = RnnModel::from_bytes(&float)
        .expect("the committed float blob must load")
        .quantized();
    let loaded = RnnModel::from_bytes(&int8).expect("the committed int8 blob must load");
    // Same quantization, so the models must be equal, not merely close — `corr` is derived
    // data, so the derived value has to agree as well as the stored bytes.
    assert_eq!(loaded, runtime);
    assert_eq!(run(&runtime, &frames), run(&loaded, &frames));
}

#[test]
fn a_corrupted_container_fails_with_both_checksums() {
    let Some(bytes) = read_or_skip(INT8_BLOB) else {
        return;
    };
    let mut corrupt = bytes.clone();
    corrupt[16] ^= 0x01; // first payload byte
    let err = RnnModel::from_bytes(&corrupt).expect_err("a corrupted payload must not load");
    assert!(matches!(
        err,
        ModelError::ChecksumMismatch { expected, actual } if expected != actual
    ));
}

#[test]
fn a_container_from_an_unknown_version_is_rejected() {
    let Some(bytes) = read_or_skip(INT8_BLOB) else {
        return;
    };
    let mut newer = bytes.clone();
    newer[4..8].copy_from_slice(&2u32.to_le_bytes());
    // The header's own checksum is over the payload only, so bumping the version is caught by
    // the version check, not by the checksum.
    assert_eq!(
        RnnModel::from_bytes(&newer).err(),
        Some(ModelError::UnsupportedVersion(2))
    );
}

#[test]
fn a_record_count_that_does_not_match_the_payload_is_rejected() {
    let Some(bytes) = read_or_skip(INT8_BLOB) else {
        return;
    };
    let mut wrong = bytes.clone();
    wrong[8..12].copy_from_slice(&7u32.to_le_bytes());
    assert_eq!(
        RnnModel::from_bytes(&wrong).err(),
        Some(ModelError::RecordCountMismatch {
            expected: 7,
            found: 30,
        })
    );
}

#[test]
fn a_bare_stream_without_a_header_still_loads_and_is_not_verified() {
    let Some(bytes) = read_or_skip(FLOAT_BLOB) else {
        return;
    };
    // Upstream `dump_weights_blob` writes no header, so a bare stream must keep loading — the
    // crate cannot retroactively check something upstream never wrote.
    assert!(RnnModel::from_bytes(&bytes).is_ok());
}

#[cfg(feature = "model-loader")]
#[test]
fn a_loader_reports_a_missing_file_and_a_pinned_mismatch() {
    let err = RnnModel::from_file(Path::new("models/nope.bin"))
        .expect_err("a missing file must not load");
    assert!(matches!(err, ModelError::Io(_)), "{err}");

    let Some(bytes) = read_or_skip(INT8_BLOB) else {
        return;
    };
    // The pinned value is checked before parsing, so a structurally plausible but wrong file is
    // refused instead of silently producing a worse model. The pin is the CRC32 of the record
    // stream, not of the file, so it is the same number for a bare blob and a container.
    let stream_crc = crc32(&bytes[16..]);
    let wrong_pin = stream_crc ^ 1;
    let path = Path::new(INT8_BLOB);
    let err = RnnModel::from_file_with_checksum(path, wrong_pin)
        .expect_err("a wrong pinned checksum must not load");
    assert!(matches!(
        err,
        ModelError::ChecksumMismatch { expected, actual } if expected == wrong_pin
            && actual == stream_crc
    ));
    assert!(RnnModel::from_file_with_checksum(path, stream_crc).is_ok());
}
