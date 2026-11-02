//! The parsed model: the immutable weight arrays shared across denoiser instances.

use crate::nnet::LinearLayer;
use crate::weights::blob;
#[cfg(any(feature = "embedded-model", feature = "embedded-model-int8"))]
use crate::weights::embedded;
use crate::weights::error::ModelError;
use crate::weights::manifest;
use crate::weights::write;

/// A parsed RNNoise model. Construct with [`RnnModel::default`] for a built-in model, with
/// [`RnnModel::from_bytes`] for any weight blob, or — with the `model-loader` feature — with
/// `RnnModel::from_file`.
#[derive(Clone, Debug, PartialEq)]
pub struct RnnModel {
    pub(crate) conv1: LinearLayer,
    pub(crate) conv2: LinearLayer,
    pub(crate) gru1_input: LinearLayer,
    pub(crate) gru1_recurrent: LinearLayer,
    pub(crate) gru2_input: LinearLayer,
    pub(crate) gru2_recurrent: LinearLayer,
    pub(crate) gru3_input: LinearLayer,
    pub(crate) gru3_recurrent: LinearLayer,
    pub(crate) dense_out: LinearLayer,
    pub(crate) vad_dense: LinearLayer,
}

impl RnnModel {
    /// Load a model from an upstream weight blob (`rnnoise_model_from_buffer`).
    ///
    /// Both shapes are accepted: a bare `"DNNw"` record stream, and the same stream inside the
    /// optional `RNNS` container, whose format version and checksum are verified here. A blob
    /// that declares a version this loader does not know, or whose payload does not hash to its
    /// header, is rejected instead of silently producing a different model.
    pub fn from_bytes(bytes: &[u8]) -> Result<RnnModel, ModelError> {
        let (payload, header) = manifest::split(bytes)?;
        let (arrays, records) = blob::parse_records(payload)?;
        if let Some(header) = header {
            if header.record_count != records {
                return Err(ModelError::RecordCountMismatch {
                    expected: header.record_count,
                    found: records,
                });
            }
        }
        // The dimensions are the model's geometry (see doc/architecture.md), one row per layer
        // in the order `init_rnnoise` wires them.
        Ok(RnnModel {
            conv1: blob::linear(&arrays, "conv1", 195, 128)?,
            conv2: blob::linear(&arrays, "conv2", 384, 384)?,
            gru1_input: blob::linear(&arrays, "gru1_input", 384, 1152)?,
            gru1_recurrent: blob::linear(&arrays, "gru1_recurrent", 384, 1152)?,
            gru2_input: blob::linear(&arrays, "gru2_input", 384, 1152)?,
            gru2_recurrent: blob::linear(&arrays, "gru2_recurrent", 384, 1152)?,
            gru3_input: blob::linear(&arrays, "gru3_input", 384, 1152)?,
            gru3_recurrent: blob::linear(&arrays, "gru3_recurrent", 384, 1152)?,
            dense_out: blob::linear(&arrays, "dense_out", 1536, 32)?,
            vad_dense: blob::linear(&arrays, "vad_dense", 1536, 1)?,
        })
    }

    /// Serialise the model as a weight blob: the `RNNS` container plus its records. The result
    /// round-trips through [`RnnModel::from_bytes`], which also re-checks the header.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        write::container(self)
    }

    /// Return an int8-quantized copy of this model: a ~4×-smaller weight
    /// footprint and meaningfully faster inference, at the cost of a small loss
    /// of numerical accuracy (it is **not** bit-exact with the C reference).
    ///
    /// Mirrors upstream's choice to quantize only conv2 and the GRU matrices;
    /// conv1 and the output heads stay full precision. Layers that are already stored as int8
    /// (a blob loaded with `embedded-model-int8`) are left as they are.
    #[must_use]
    pub fn quantized(mut self) -> RnnModel {
        for layer in [
            &mut self.conv2,
            &mut self.gru1_input,
            &mut self.gru1_recurrent,
            &mut self.gru2_input,
            &mut self.gru2_recurrent,
            &mut self.gru3_input,
            &mut self.gru3_recurrent,
        ] {
            layer.quantize();
        }
        self
    }
}

#[cfg(any(feature = "embedded-model", feature = "embedded-model-int8"))]
impl Default for RnnModel {
    /// The built-in model: the float blob when `embedded-model` is on, otherwise the
    /// pre-quantized blob from `embedded-model-int8`. When both are enabled the float blob
    /// wins, because feature unification can enable both without anyone asking.
    ///
    /// # Panics
    /// Panics only if an embedded blob is invalid — a build-time problem, not a runtime one:
    /// the committed artifacts are checked by `tests/model_artifacts.rs`.
    fn default() -> RnnModel {
        #[cfg(feature = "embedded-model")]
        let bytes = embedded::FLOAT_BLOB;
        #[cfg(not(feature = "embedded-model"))]
        let bytes = embedded::INT8_BLOB;
        RnnModel::from_bytes(bytes).expect("embedded default model is valid")
    }
}

#[cfg(all(test, feature = "embedded-model"))]
mod tests {
    use super::*;

    #[test]
    fn default_model_gru_layers_are_dense_equivalent() {
        // The shipped model's GRU indices are complete and ascending, so the layers load
        // in the dense form; the parity test proves the output is unchanged.
        let model = RnnModel::default();
        assert!(model.gru1_input.weights_idx.is_none());
        assert_eq!(
            model.gru1_input.nb_inputs * model.gru1_input.nb_outputs,
            384 * 1152
        );
    }

    #[test]
    fn default_model_layers_use_the_block_form_where_possible() {
        let model = RnnModel::default();
        for layer in [
            &model.conv1,
            &model.conv2,
            &model.gru1_input,
            &model.gru1_recurrent,
            &model.gru2_input,
            &model.gru2_recurrent,
            &model.gru3_input,
            &model.gru3_recurrent,
            &model.dense_out,
        ] {
            assert_eq!(
                layer.dense_layout,
                crate::nnet::DenseLayout::Blocks8,
                "layer with {} outputs should be in block form",
                layer.nb_outputs
            );
        }
        // `vad_dense` has a single output, so it cannot form a group of eight.
        assert_eq!(
            model.vad_dense.dense_layout,
            crate::nnet::DenseLayout::RowMajor
        );
    }
}
