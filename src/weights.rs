//! Model loading: parses the upstream RNNoise binary weight-blob format
//! (`"DNNw"` records, produced by `dump_weights_blob` / `write_weights.c`) and
//! wires the named arrays into the 10-layer [`RnnModel`]. The default model is
//! embedded at compile time.

use std::collections::HashMap;
use std::fmt;

use crate::nnet::{LinearLayer, Weights};

/// The default RNNoise model (float weights), embedded in the binary.
static DEFAULT_MODEL_BLOB: &[u8] = include_bytes!("../models/rnnoise_default.bin");

const WEIGHT_BLOCK_SIZE: usize = 64;
const WEIGHT_TYPE_FLOAT: i32 = 0;
const WEIGHT_TYPE_INT: i32 = 1;

/// Errors returned while loading a model blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// The blob is truncated or otherwise structurally invalid.
    Malformed,
    /// A required weight array is missing.
    MissingArray(String),
    /// A weight array has an unexpected length.
    BadSize(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Malformed => write!(f, "malformed model blob"),
            ModelError::MissingArray(n) => write!(f, "missing weight array `{n}`"),
            ModelError::BadSize(n) => write!(f, "weight array `{n}` has an unexpected size"),
        }
    }
}

impl std::error::Error for ModelError {}

/// A parsed RNNoise model: the immutable weights shared across denoiser
/// instances. Construct with [`RnnModel::default`] for the built-in model, or
/// [`RnnModel::from_bytes`] to load a custom one.
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

struct Array<'a> {
    kind: i32,
    data: &'a [u8],
}

/// Parse the `"DNNw"` record stream into a name → array map (`parse_weights`).
fn parse_weights(blob: &[u8]) -> Result<HashMap<&str, Array<'_>>, ModelError> {
    let mut map = HashMap::new();
    let mut off = 0usize;
    while off < blob.len() {
        if off + WEIGHT_BLOCK_SIZE > blob.len() {
            return Err(ModelError::Malformed);
        }
        let head = &blob[off..off + WEIGHT_BLOCK_SIZE];
        if &head[0..4] != b"DNNw" {
            return Err(ModelError::Malformed);
        }
        let kind = i32::from_le_bytes([head[8], head[9], head[10], head[11]]);
        let size = i32::from_le_bytes([head[12], head[13], head[14], head[15]]);
        let block_size = i32::from_le_bytes([head[16], head[17], head[18], head[19]]);
        if size < 0 || block_size < size {
            return Err(ModelError::Malformed);
        }
        let (size, block_size) = (size as usize, block_size as usize);
        // name[44] starts at byte 20, NUL-terminated.
        let name_bytes = &head[20..64];
        let nlen = name_bytes.iter().position(|&b| b == 0).unwrap_or(44);
        let name = std::str::from_utf8(&name_bytes[..nlen]).map_err(|_| ModelError::Malformed)?;

        let data_start = off + WEIGHT_BLOCK_SIZE;
        if data_start + block_size > blob.len() {
            return Err(ModelError::Malformed);
        }
        map.insert(
            name,
            Array {
                kind,
                data: &blob[data_start..data_start + size],
            },
        );
        off = data_start + block_size;
    }
    Ok(map)
}

fn f32_array(arrays: &HashMap<&str, Array<'_>>, name: &str) -> Result<Vec<f32>, ModelError> {
    let a = arrays
        .get(name)
        .ok_or_else(|| ModelError::MissingArray(name.to_string()))?;
    if a.kind != WEIGHT_TYPE_FLOAT || a.data.len() % 4 != 0 {
        return Err(ModelError::BadSize(name.to_string()));
    }
    Ok(a.data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn i32_array(arrays: &HashMap<&str, Array<'_>>, name: &str) -> Result<Vec<i32>, ModelError> {
    let a = arrays
        .get(name)
        .ok_or_else(|| ModelError::MissingArray(name.to_string()))?;
    if a.kind != WEIGHT_TYPE_INT || a.data.len() % 4 != 0 {
        return Err(ModelError::BadSize(name.to_string()));
    }
    Ok(a.data
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Build a dense or sparse layer, validating bias/weight sizes (`linear_init`).
fn linear(
    arrays: &HashMap<&str, Array<'_>>,
    bias: &str,
    weights: &str,
    idx: Option<&str>,
    diag: Option<&str>,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<LinearLayer, ModelError> {
    let bias_v = f32_array(arrays, bias)?;
    if bias_v.len() != nb_outputs {
        return Err(ModelError::BadSize(bias.to_string()));
    }
    let float_weights = f32_array(arrays, weights)?;
    let weights_idx = idx.map(|n| i32_array(arrays, n)).transpose()?;
    let diag_v = diag.map(|n| f32_array(arrays, n)).transpose()?;

    // Size checks mirroring `linear_init`.
    match &weights_idx {
        None => {
            if float_weights.len() != nb_inputs * nb_outputs {
                return Err(ModelError::BadSize(weights.to_string()));
            }
        }
        Some(idxv) => {
            let total_blocks = sparse_total_blocks(idxv, nb_outputs)
                .ok_or_else(|| ModelError::BadSize(idx.unwrap().to_string()))?;
            if float_weights.len() != 32 * total_blocks {
                return Err(ModelError::BadSize(weights.to_string()));
            }
        }
    }
    if let Some(d) = &diag_v {
        if d.len() != nb_outputs {
            return Err(ModelError::BadSize(diag.unwrap().to_string()));
        }
    }

    Ok(LinearLayer {
        bias: bias_v,
        weights: Weights::Float(float_weights),
        weights_idx,
        diag: diag_v,
        nb_inputs,
        nb_outputs,
    })
}

/// Count the total number of 8×4 weight blocks described by a sparse `idx`
/// array, validating its structure (`find_idx_check`).
fn sparse_total_blocks(idx: &[i32], nb_outputs: usize) -> Option<usize> {
    let mut remain = idx.len() as i64;
    let mut pos = 0usize;
    let mut nb_out = nb_outputs as i64;
    let mut total = 0usize;
    while remain > 0 {
        let nb_blocks = *idx.get(pos)? as i64;
        if nb_blocks < 0 || remain < nb_blocks + 1 {
            return None;
        }
        pos += 1 + nb_blocks as usize;
        nb_out -= 8;
        remain -= nb_blocks + 1;
        total += nb_blocks as usize;
    }
    if nb_out != 0 {
        return None;
    }
    Some(total)
}

impl RnnModel {
    /// Load a model from an upstream weight-blob (`rnnoise_model_from_buffer`).
    pub fn from_bytes(blob: &[u8]) -> Result<RnnModel, ModelError> {
        let arrays = parse_weights(blob)?;
        Ok(RnnModel {
            conv1: linear(
                &arrays,
                "conv1_bias",
                "conv1_weights_float",
                None,
                None,
                195,
                128,
            )?,
            conv2: linear(
                &arrays,
                "conv2_bias",
                "conv2_weights_float",
                None,
                None,
                384,
                384,
            )?,
            gru1_input: linear(
                &arrays,
                "gru1_input_bias",
                "gru1_input_weights_float",
                Some("gru1_input_weights_idx"),
                None,
                384,
                1152,
            )?,
            gru1_recurrent: linear(
                &arrays,
                "gru1_recurrent_bias",
                "gru1_recurrent_weights_float",
                Some("gru1_recurrent_weights_idx"),
                Some("gru1_recurrent_weights_diag"),
                384,
                1152,
            )?,
            gru2_input: linear(
                &arrays,
                "gru2_input_bias",
                "gru2_input_weights_float",
                Some("gru2_input_weights_idx"),
                None,
                384,
                1152,
            )?,
            gru2_recurrent: linear(
                &arrays,
                "gru2_recurrent_bias",
                "gru2_recurrent_weights_float",
                Some("gru2_recurrent_weights_idx"),
                Some("gru2_recurrent_weights_diag"),
                384,
                1152,
            )?,
            gru3_input: linear(
                &arrays,
                "gru3_input_bias",
                "gru3_input_weights_float",
                Some("gru3_input_weights_idx"),
                None,
                384,
                1152,
            )?,
            gru3_recurrent: linear(
                &arrays,
                "gru3_recurrent_bias",
                "gru3_recurrent_weights_float",
                Some("gru3_recurrent_weights_idx"),
                Some("gru3_recurrent_weights_diag"),
                384,
                1152,
            )?,
            dense_out: linear(
                &arrays,
                "dense_out_bias",
                "dense_out_weights_float",
                None,
                None,
                1536,
                32,
            )?,
            vad_dense: linear(
                &arrays,
                "vad_dense_bias",
                "vad_dense_weights_float",
                None,
                None,
                1536,
                1,
            )?,
        })
    }

    /// Return an int8-quantized copy of this model: a ~4×-smaller weight
    /// footprint and meaningfully faster inference, at the cost of a small loss
    /// of numerical accuracy (it is **not** bit-exact with the C reference).
    ///
    /// Mirrors upstream's choice to quantize only conv2 and the GRU matrices;
    /// conv1 and the output heads stay full precision.
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

impl Default for RnnModel {
    /// The built-in RNNoise model.
    fn default() -> RnnModel {
        RnnModel::from_bytes(DEFAULT_MODEL_BLOB).expect("embedded default model is valid")
    }
}
