//! Model loading: the `"DNNw"` weight blob, its optional integrity container, the loaders,
//! the embedded artifacts, and the parsed [`RnnModel`].
//!
//! This module root only declares submodules and re-exports the public surface (no logic or
//! types in a module root).

pub(crate) mod blob;
pub(crate) mod embedded;
pub(crate) mod error;
#[cfg(feature = "model-loader")]
mod loader;
pub(crate) mod manifest;
pub(crate) mod model;
pub(crate) mod write;

pub use error::ModelError;
pub use manifest::crc32;
pub use model::RnnModel;
