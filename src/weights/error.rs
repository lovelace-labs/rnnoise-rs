//! The error type shared by the blob parser, the integrity container and the loaders.

use std::fmt;

/// Errors returned while loading a model blob.
///
/// `#[non_exhaustive]` so a new failure mode can be added without breaking a caller that
/// matches the known variants.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelError {
    /// The blob is truncated or otherwise structurally invalid.
    Malformed,
    /// A required weight array is missing.
    MissingArray(String),
    /// A weight array has an unexpected length.
    BadSize(String),
    /// A weight array has an unexpected record type.
    BadType(String),
    /// A record, or the `RNNS` container, declares a version this loader does not understand.
    UnsupportedVersion(u32),
    /// The record stream does not hash to the expected checksum — either the container header's
    /// own checksum, or a value the caller pinned.
    ChecksumMismatch {
        /// Checksum that was expected.
        expected: u32,
        /// CRC32 of the record stream as read.
        actual: u32,
    },
    /// The container header declares a record count that the payload does not contain.
    RecordCountMismatch {
        /// Record count declared by the header.
        expected: u32,
        /// Record count actually parsed.
        found: u32,
    },
    /// The crate was built without an embedded model, so there is no default model to use.
    NoEmbeddedModel,
    /// A loader could not read its source. The message names the path and the operating-system
    /// failure.
    Io(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Malformed => write!(f, "malformed model blob"),
            ModelError::MissingArray(n) => write!(f, "missing weight array `{n}`"),
            ModelError::BadSize(n) => write!(f, "weight array `{n}` has an unexpected size"),
            ModelError::BadType(n) => write!(f, "weight array `{n}` has an unexpected type"),
            ModelError::UnsupportedVersion(v) => write!(f, "unsupported model blob version {v}"),
            ModelError::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected:#010x}, found {actual:#010x}")
            }
            ModelError::RecordCountMismatch { expected, found } => {
                write!(f, "container declares {expected} records, found {found}")
            }
            ModelError::Io(m) => write!(f, "{m}"),
            ModelError::NoEmbeddedModel => write!(
                f,
                "no embedded model: build with `embedded-model` or `embedded-model-int8`, or load one"
            ),
        }
    }
}

impl std::error::Error for ModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_name_what_was_wrong() {
        // The point of these variants is that a caller can log *which* array or which checksum
        // failed, not just that loading failed.
        assert_eq!(
            ModelError::BadType("gru1_input_weights_float".to_string()).to_string(),
            "weight array `gru1_input_weights_float` has an unexpected type"
        );
        assert_eq!(
            ModelError::ChecksumMismatch {
                expected: 0x89d3_20c8,
                actual: 0x2d_f8_ef_d6,
            }
            .to_string(),
            "checksum mismatch: expected 0x89d320c8, found 0x2df8efd6"
        );
    }
}
