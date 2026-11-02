//! The weight blobs that are compiled into the crate.
//!
//! Each blob is only compiled in when its feature is on, so a consumer that links the crate
//! without an embedded model pays for code, not for 11.3 MiB of data: rustc drops the blob
//! when nothing references it, and the `Default` impl below is the only thing that ever does.

#[cfg(feature = "embedded-model")]
pub(crate) const FLOAT_BLOB: &[u8] = include_bytes!("../../models/rnnoise_default.bin");

#[cfg(all(feature = "embedded-model-int8", not(feature = "embedded-model")))]
pub(crate) const INT8_BLOB: &[u8] = include_bytes!("../../models/rnnoise_default_int8.bin");
