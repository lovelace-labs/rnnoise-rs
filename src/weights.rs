//! Model loading: parses the upstream RNNoise binary weight-blob format
//! (`"DNNw"` records, produced by `dump_weights_blob` / `write_weights.c`) and
//! wires the named arrays into the 10-layer [`RnnModel`]. The default model is
//! embedded at compile time.

use std::collections::HashMap;
use std::fmt;

use crate::nnet::{DenseLayout, LinearLayer, Weights};

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
        .as_chunks::<4>()
        .0
        .iter()
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The model stores the GRU matrices in the sparse 8×4 layout. When the index is
/// *complete and ascending* — every row group lists `nb_inputs / 4` blocks at positions
/// `0, 4, 8, …` — the sparse walk visits each output's contributions in ascending input
/// order, which is exactly `dense_sgemv`'s accumulation order. Densifying such a layer
/// therefore cannot change a single bit; it just lets the GEMV run through the wider
/// tier (measured: the GRU layers go from ~40 to ~65 GB/s on a Zen 3 core).
///
/// Genuinely sparse or out-of-order indices are left on the sparse kernel, so the
/// accumulation order is always the one the blob describes.
fn complete_ascending_dense_form(idx: &[i32], w: &[f32], m: usize, n: usize) -> Option<Vec<f32>> {
    if !m.is_multiple_of(4) || !n.is_multiple_of(8) {
        return None;
    }
    let mut dense = vec![0.0f32; n * m];
    let mut wi = 0usize;
    let mut ii = 0usize;
    let mut row = 0usize;
    while row < n {
        let cols = idx[ii] as usize;
        ii += 1;
        if cols * 4 != m {
            return None;
        }
        for k in 0..cols {
            let pos = idx[ii] as usize;
            ii += 1;
            if pos != k * 4 {
                return None;
            }
            for c in 0..4 {
                for r in 0..8 {
                    dense[(pos + c) * n + row + r] = w[wi + c * 8 + r];
                }
            }
            wi += 32;
        }
        row += 8;
    }
    Some(dense)
}

/// The same `n * m` floats, permuted into the block-major form that
/// [`crate::simd::dense_sgemv_blocks`] expects: one contiguous run of `m` values per group
/// of eight outputs, plus a run per output for any incomplete last group.
fn block_major_form(dense: &[f32], m: usize, n: usize) -> Vec<f32> {
    let groups = n / 8;
    let mut out = vec![0.0f32; dense.len()];
    for j in 0..m {
        for o in 0..n {
            let dst = if o < groups * 8 {
                (o / 8) * 8 * m + j * 8 + (o % 8)
            } else {
                groups * 8 * m + (o - groups * 8) * m + j
            };
            out[dst] = dense[j * n + o];
        }
    }
    out
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

    // A complete, ascending sparse index is dense-equivalent: same values, same
    // accumulation order per output, so the fast GEMV tier can be used.
    let dense = weights_idx.as_ref().and_then(|idxv| {
        complete_ascending_dense_form(idxv, &float_weights, nb_inputs, nb_outputs)
    });
    let is_dense = dense.is_some() || weights_idx.is_none();
    // A dense layer whose outputs all sit in complete groups can be stored in the
    // block-major form, which is what lets the tier keep accumulators in registers.
    let use_blocks = is_dense && nb_outputs.is_multiple_of(8);
    let stored = match (dense, use_blocks) {
        (Some(d), true) => block_major_form(&d, nb_inputs, nb_outputs),
        (Some(d), false) => d,
        (None, true) => block_major_form(&float_weights, nb_inputs, nb_outputs),
        (None, false) => float_weights,
    };

    Ok(LinearLayer {
        bias: bias_v,
        weights: Weights::Float(stored),
        weights_idx: if is_dense { None } else { weights_idx },
        diag: diag_v,
        nb_inputs,
        nb_outputs,
        dense_layout: if use_blocks {
            DenseLayout::Blocks8
        } else {
            DenseLayout::RowMajor
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_ascending_index_is_dense_equivalent() {
        // One row group, every column block present in order.
        let (m, n) = (16usize, 8usize);
        let idx = vec![4, 0, 4, 8, 12];
        let w: Vec<f32> = (0..4 * 32).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let x: Vec<f32> = (0..m).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let dense = complete_ascending_dense_form(&idx, &w, m, n).expect("complete index");
        let mut sparse_out = vec![0.0f32; n];
        let mut dense_out = vec![0.0f32; n];
        crate::simd::sparse_sgemv8x4(&mut sparse_out, &w, &idx, n, &x);
        crate::simd::dense_sgemv(&mut dense_out, &dense, m, n, &x);
        assert_eq!(
            sparse_out, dense_out,
            "densifying changed the accumulation order"
        );
    }

    #[test]
    fn incomplete_or_unordered_index_stays_sparse() {
        let (m, n) = (16usize, 8usize);
        let w = vec![0.1f32; 3 * 32];
        assert!(complete_ascending_dense_form(&[3, 0, 4, 8], &w, m, n).is_none());
        let w4 = vec![0.1f32; 4 * 32];
        assert!(complete_ascending_dense_form(&[4, 0, 8, 4, 12], &w4, m, n).is_none());
    }

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
}

#[cfg(test)]
mod block_form_tests {
    use super::*;

    #[test]
    fn block_form_is_only_a_permutation() {
        let (m, n) = (16usize, 24usize);
        let mut s = 17u32;
        let mut rng = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32) / 16_777_216.0 - 0.5
        };
        let row: Vec<f32> = (0..m * n).map(|_| rng()).collect();
        let block = block_major_form(&row, m, n);

        let mut a = row.clone();
        let mut b = block.clone();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(a, b, "block form must hold exactly the same values");

        // ... and the two kernels must therefore produce the same bits.
        let x: Vec<f32> = (0..m).map(|_| rng()).collect();
        let mut o1 = vec![0.0f32; n];
        let mut o2 = vec![0.0f32; n];
        crate::simd::dense_sgemv(&mut o1, &row, m, n, &x);
        crate::simd::dense_sgemv_blocks(&mut o2, &block, m, n, &x);
        assert_eq!(o1, o2, "block form changed a bit");
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
                DenseLayout::Blocks8,
                "layer with {} outputs should be in block form",
                layer.nb_outputs
            );
        }
        // `vad_dense` has a single output, so it cannot form a group of eight.
        assert_eq!(model.vad_dense.dense_layout, DenseLayout::RowMajor);
    }
}
