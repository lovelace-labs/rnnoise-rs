//! Writing a model back to a weight blob — the mirror of [`crate::weights::blob`].
//!
//! The writer is what makes the shipped int8 artifact reproducible inside the crate: nothing
//! here re-implements the quantization, it serialises whatever the model already holds.
//! `from_bytes(to_bytes(m)) == m` is asserted below, and `tests/model_artifacts.rs` asserts that
//! the committed int8 blob is exactly what this writer produces from the committed float blob.

use crate::nnet::{DenseLayout, LinearLayer, Weights};
use crate::weights::manifest;
use crate::weights::model::RnnModel;

const RECORD_HEADER_SIZE: usize = 64;

const RECORD_TYPE_FLOAT: i32 = 0;
const RECORD_TYPE_INT8: i32 = 3;

/// Serialise a model as the `RNNS` container plus its records.
pub(crate) fn container(model: &RnnModel) -> Vec<u8> {
    // The same layer order as `init_rnnoise` (and as `model::from_bytes`), so writer and reader
    // agree on the names without a lookup table.
    let layers: [(&str, &LinearLayer); 10] = [
        ("conv1", &model.conv1),
        ("conv2", &model.conv2),
        ("gru1_input", &model.gru1_input),
        ("gru1_recurrent", &model.gru1_recurrent),
        ("gru2_input", &model.gru2_input),
        ("gru2_recurrent", &model.gru2_recurrent),
        ("gru3_input", &model.gru3_input),
        ("gru3_recurrent", &model.gru3_recurrent),
        ("dense_out", &model.dense_out),
        ("vad_dense", &model.vad_dense),
    ];

    let mut payload = Vec::new();
    for (base, layer) in layers {
        match &layer.weights {
            Weights::Float(w) => {
                // `Weights::Float` can hold the block-permuted form the GEMV tier wants; the blob
                // format is row-major, so undo the permutation here.
                let row_major = if layer.dense_layout == DenseLayout::Blocks8 {
                    row_major_form(w, layer.nb_inputs, layer.nb_outputs)
                } else {
                    w.clone()
                };
                emit(
                    &mut payload,
                    &format!("{base}_weights_float"),
                    RECORD_TYPE_FLOAT,
                    &le_bytes(&row_major),
                );
            }
            Weights::Q8 { w, scale, .. } => {
                emit(
                    &mut payload,
                    &format!("{base}_weights_int8"),
                    RECORD_TYPE_INT8,
                    &as_bytes(w),
                );
                emit(
                    &mut payload,
                    &format!("{base}_scale"),
                    RECORD_TYPE_FLOAT,
                    &le_bytes(scale),
                );
            }
        }
        if let Some(diag) = &layer.diag {
            emit(
                &mut payload,
                &format!("{base}_weights_diag"),
                RECORD_TYPE_FLOAT,
                &le_bytes(diag),
            );
        }
        emit(
            &mut payload,
            &format!("{base}_bias"),
            RECORD_TYPE_FLOAT,
            &le_bytes(&layer.bias),
        );
    }

    let records: u32 = layers
        .iter()
        .map(|(_, layer)| {
            2 + usize::from(layer.diag.is_some())
                + usize::from(matches!(layer.weights, Weights::Q8 { .. }))
        })
        .sum::<usize>() as u32;
    let mut out = Vec::with_capacity(manifest::HEADER_SIZE + payload.len());
    out.extend_from_slice(manifest::CONTAINER_MAGIC);
    out.extend_from_slice(&manifest::FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&records.to_le_bytes());
    out.extend_from_slice(&manifest::crc32(&payload).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// One `"DNNw"` record: header, payload, then padding to the next multiple of
/// `RECORD_HEADER_SIZE` — upstream pads records so a reader can skip by `block_size`.
fn emit(out: &mut Vec<u8>, name: &str, kind: i32, data: &[u8]) {
    let block_size = (data.len().div_ceil(RECORD_HEADER_SIZE) * RECORD_HEADER_SIZE) as i32;
    let mut head = vec![0u8; RECORD_HEADER_SIZE];
    head[..4].copy_from_slice(b"DNNw");
    head[4..8].copy_from_slice(&0i32.to_le_bytes()); // record version
    head[8..12].copy_from_slice(&kind.to_le_bytes());
    head[12..16].copy_from_slice(&(data.len() as i32).to_le_bytes());
    head[16..20].copy_from_slice(&block_size.to_le_bytes());
    head[20..20 + name.len()].copy_from_slice(name.as_bytes());
    out.extend_from_slice(&head);
    out.extend_from_slice(data);
    out.resize(out.len() + (block_size - data.len() as i32) as usize, 0);
}

/// int8 weights as raw bytes (two's complement, machine-endian, as upstream stores them).
fn as_bytes(values: &[i8]) -> Vec<u8> {
    values.iter().map(|&v| v as u8).collect()
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Undo [`crate::weights::blob::block_major_form`]: the block-permuted storage back to the
/// row-major order the blob format stores.
fn row_major_form(block: &[f32], m: usize, n: usize) -> Vec<f32> {
    let groups = n / 8;
    let mut out = vec![0.0f32; block.len()];
    for j in 0..m {
        for o in 0..n {
            let src = if o < groups * 8 {
                (o / 8) * 8 * m + j * 8 + (o % 8)
            } else {
                groups * 8 * m + (o - groups * 8) * m + j
            };
            out[j * n + o] = block[src];
        }
    }
    out
}

#[cfg(all(test, feature = "embedded-model"))]
mod tests {
    use super::*;
    use crate::weights::blob;

    #[test]
    fn the_model_round_trips_through_the_writer() {
        // The writer must reproduce exactly what the reader expects: names, types, sizes, and the
        // row-major order the blob format stores (the model itself keeps the block-permuted form
        // the GEMV tier wants).
        let model = RnnModel::default();
        let back = RnnModel::from_bytes(&container(&model)).expect("the writer must be readable");
        assert_eq!(back, model);
    }

    #[test]
    fn the_writer_reproduces_the_block_permutation_it_wrote() {
        let (m, n) = (16usize, 24usize);
        let row: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let block = blob::block_major_form(&row, m, n);
        assert_eq!(row_major_form(&block, m, n), row);
    }
}
