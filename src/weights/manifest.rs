//! The optional integrity container around a weight blob.
//!
//! A blob is either a bare stream of upstream `"DNNw"` records (what
//! `dump_weights_blob` writes), or that same stream wrapped in a small header:
//!
//! ```text
//! magic             4 bytes  "RNNS"
//! format version    4 bytes  u32 (1)
//! record count      4 bytes  u32
//! payload checksum  4 bytes  u32 (CRC32, IEEE reflected, over the payload only)
//! payload           n bytes  the unchanged "DNNw" record stream
//! ```
//!
//! The header is what turns a corrupt or unknown blob into a load-time error
//! instead of a silently worse model: a bare stream can only be checked
//! structurally (names, sizes, record types), so a truncated download with
//! plausible-looking headers still parses and the denoiser just sounds worse.
//! Callers that download weights pin a checksum with [`crc32`] and compare it
//! before loading; [`split`] verifies the header's own checksum when present.

use crate::weights::error::ModelError;

/// Magic bytes of the container header.
pub(crate) const CONTAINER_MAGIC: &[u8; 4] = b"RNNS";

/// Container format version written by this crate. A newer version is rejected,
/// never guessed.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// Header size: magic + format version + record count + checksum.
pub(crate) const HEADER_SIZE: usize = 16;

/// Parsed container header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Manifest {
    pub(crate) format_version: u32,
    pub(crate) record_count: u32,
    pub(crate) checksum: u32,
}

/// CRC32 (IEEE, reflected) — table-free pure Rust, so the core library stays
/// dependency-free. This is a corruption check, not an authenticity check.
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    const POLYNOMIAL: u32 = 0xedb8_8320;
    let mut crc = !0u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            // `!((crc & 1) - 1)` is all ones when the low bit is set, zero otherwise.
            crc = (crc >> 1) ^ (POLYNOMIAL & !((crc & 1).wrapping_sub(1)));
        }
    }
    !crc
}

/// Compare a blob against a checksum the caller pinned, before anything is parsed.
///
/// The value is always the CRC32 of the record stream, so it is the same number for a bare
/// stream and for a container: a pinned checksum also catches a blob that is structurally
/// plausible but not the model the application expected.
#[cfg(feature = "model-loader")]
pub(crate) fn verify(bytes: &[u8], expected: u32) -> Result<(), ModelError> {
    let (payload, _) = split(bytes)?;
    let actual = crc32(payload);
    if actual != expected {
        return Err(ModelError::ChecksumMismatch { expected, actual });
    }
    Ok(())
}

/// Split a blob into payload and header, verifying the header checksum when a
/// container is present. A bare record stream yields `manifest == None`.
pub(crate) fn split(blob: &[u8]) -> Result<(&[u8], Option<Manifest>), ModelError> {
    if blob[..4.min(blob.len())] != *CONTAINER_MAGIC {
        return Ok((blob, None));
    }
    if blob.len() < HEADER_SIZE {
        return Err(ModelError::Malformed);
    }
    let format_version = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]);
    let record_count = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
    let checksum = u32::from_le_bytes([blob[12], blob[13], blob[14], blob[15]]);
    let payload = &blob[HEADER_SIZE..];
    if format_version != FORMAT_VERSION {
        return Err(ModelError::UnsupportedVersion(format_version));
    }
    let actual = crc32(payload);
    if actual != checksum {
        return Err(ModelError::ChecksumMismatch {
            expected: checksum,
            actual,
        });
    }
    Ok((
        payload,
        Some(Manifest {
            format_version,
            record_count,
            checksum,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One `"DNNw"` record, so the container tests exercise a real payload.
    fn record(name: &str, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; HEADER_SIZE_RECORD + payload.len()];
        out[..4].copy_from_slice(b"DNNw");
        out[4..8].copy_from_slice(&0i32.to_le_bytes());
        out[8..12].copy_from_slice(&0i32.to_le_bytes());
        out[12..16].copy_from_slice(&(payload.len() as i32).to_le_bytes());
        out[16..20].copy_from_slice(&(payload.len() as i32).to_le_bytes());
        out[20..20 + name.len()].copy_from_slice(name.as_bytes());
        out[20 + name.len()..64].fill(0);
        out[64..].copy_from_slice(payload);
        out
    }

    const HEADER_SIZE_RECORD: usize = 64;

    fn container(record_count: u32, payload: &[u8], version: u32) -> Vec<u8> {
        let mut out = vec![0u8; HEADER_SIZE + payload.len()];
        out[..4].copy_from_slice(CONTAINER_MAGIC);
        out[4..8].copy_from_slice(&version.to_le_bytes());
        out[8..12].copy_from_slice(&record_count.to_le_bytes());
        out[12..16].copy_from_slice(&crc32(payload).to_le_bytes());
        out[HEADER_SIZE..].copy_from_slice(payload);
        out
    }

    #[test]
    fn crc32_matches_the_reference_vectors() {
        // The published CRC32 check value, plus a second independent vector.
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414f_a339
        );
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn a_bare_stream_is_accepted_without_a_header() {
        let blob = record("conv1_bias", &1.0f32.to_le_bytes());
        let (payload, manifest) = split(&blob).expect("bare stream");
        assert_eq!(payload, &blob[..]);
        assert_eq!(manifest, None);
    }

    #[test]
    fn a_valid_container_verifies() {
        let blob = container(
            1,
            &record("conv1_bias", &1.0f32.to_le_bytes()),
            FORMAT_VERSION,
        );
        let (_, manifest) = split(&blob).expect("valid container");
        assert_eq!(
            manifest,
            Some(Manifest {
                format_version: FORMAT_VERSION,
                record_count: 1,
                checksum: crc32(&blob[HEADER_SIZE..]),
            })
        );
    }

    #[test]
    fn a_corrupted_payload_reports_both_checksums() {
        let payload = record("conv1_bias", &1.0f32.to_le_bytes());
        let mut blob = container(1, &payload, FORMAT_VERSION);
        blob[HEADER_SIZE] ^= 0x01;
        let err = split(&blob).expect_err("a corrupted payload must not load");
        assert!(matches!(
            err,
            ModelError::ChecksumMismatch { expected, actual } if expected != actual
        ));
    }

    #[test]
    fn an_unknown_container_version_is_rejected() {
        let err = split(&container(
            1,
            &record("conv1_bias", &1.0f32.to_le_bytes()),
            2,
        ))
        .expect_err("version 2 is unknown to this loader");
        assert_eq!(err, ModelError::UnsupportedVersion(2));
    }

    #[test]
    fn a_container_shorter_than_its_header_is_malformed() {
        // The magic matches, so this is the structural failure, not a checksum one.
        assert_eq!(
            split(b"RNNS\x01\x00\x00\x00").err(),
            Some(ModelError::Malformed)
        );
    }
}
