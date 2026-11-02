//! Loading a model from disk (the `model-loader` feature).
//!
//! The file path is the on-demand half of model loading: an application that ships no embedded
//! blob can download or cache a weight blob and load it at startup. The loaders are safe for
//! that, and deliberately not for the audio thread — they allocate and perform I/O, so a
//! `DenoiseState` must be built from the returned model with
//! [`DenoiseState::new_with_model`](crate::DenoiseState::new_with_model), never per frame.
//!
//! An existing memory mapping works as-is: `RnnModel::from_bytes(&mmap)` takes any `&[u8]`, so a
//! caller that maps the file itself needs no extra API and no dependency here.

use std::path::Path;

use crate::weights::error::ModelError;
use crate::weights::manifest;
use crate::weights::model::RnnModel;

impl RnnModel {
    /// Load a model from a weight blob on disk (`rnnoise_model_from_filename`).
    ///
    /// A blob carrying the `RNNS` container header is version- and checksum-verified by
    /// [`RnnModel::from_bytes`]; a bare `"DNNw"` stream can only be checked structurally, so a
    /// truncated download with plausible headers still loads. For a file that came from a
    /// download, use [`RnnModel::from_file_with_checksum`] instead.
    ///
    /// Errors of kind [`ModelError::Io`] name the path and the operating-system failure.
    pub fn from_file(path: &Path) -> Result<RnnModel, ModelError> {
        let bytes = read(path)?;
        RnnModel::from_bytes(&bytes)
    }

    /// Load a model from disk and compare the record stream against a checksum the caller
    /// pinned, before parsing anything.
    ///
    /// The pinned value is the CRC32 of the record stream, so it is the same number whether the
    /// file carries the `RNNS` header or not. A pinned checksum is what catches a truncated or
    /// substituted download that a structural check cannot: a blob with the right names and the
    /// right sizes would otherwise load and simply sound worse. It is not an authenticity check —
    /// it says the bytes match the expected ones, not who produced them.
    pub fn from_file_with_checksum(path: &Path, expected_crc: u32) -> Result<RnnModel, ModelError> {
        let bytes = read(path)?;
        manifest::verify(&bytes, expected_crc)?;
        RnnModel::from_bytes(&bytes)
    }
}

fn read(path: &Path) -> Result<Vec<u8>, ModelError> {
    std::fs::read(path)
        .map_err(|e| ModelError::Io(format!("could not read `{}`: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_is_an_io_error_that_names_the_path() {
        let err = RnnModel::from_file(Path::new("models/definitely-not-there.bin"))
            .expect_err("a missing file must not load");
        assert!(matches!(err, ModelError::Io(_)), "{err}");
    }

    #[test]
    fn a_pinned_checksum_that_does_not_match_is_reported_with_both_values() {
        // The smallest possible record stream: one `"DNNw"` record with no payload.
        let mut record = vec![0u8; 64];
        record[..4].copy_from_slice(b"DNNw");
        record[8..12].copy_from_slice(&0i32.to_le_bytes());
        record[12..16].copy_from_slice(&0i32.to_le_bytes());
        record[16..20].copy_from_slice(&0i32.to_le_bytes());
        record[20..30].copy_from_slice(b"conv1_bias");
        let err = manifest::verify(&record, 0xdead_beef).expect_err("pinned checksum must fail");
        assert!(
            matches!(
                err,
                ModelError::ChecksumMismatch { expected, actual } if expected == 0xdead_beef
                    && actual == manifest::crc32(&record)
            ),
            "{err}"
        );
    }
}
