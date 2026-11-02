//! The `"DNNw"` record stream: parsing records (`parse_weights`) and building one
//! layer from them (`linear_init` / `find_idx_check`).

use std::collections::HashMap;

use crate::nnet::{DenseLayout, LinearLayer, Weights};
use crate::weights::error::ModelError;

/// Record header size (`WeightHead` in upstream `nnet.h`).
const RECORD_HEADER_SIZE: usize = 64;

/// Record version written by upstream `dump_weights_blob`.
const RECORD_VERSION: i32 = 0;

/// Record types (upstream `WEIGHT_TYPE_*`).
const WEIGHT_TYPE_FLOAT: i32 = 0;
const WEIGHT_TYPE_INT32: i32 = 1;
const WEIGHT_TYPE_INT8: i32 = 3;

/// A named weight array as it appears in the blob.
pub(crate) struct Array<'a> {
    kind: i32,
    data: &'a [u8],
}

/// Parse the record stream into a name -> array map. The record count is returned
/// too, so a container header can be cross-checked against what was actually found.
pub(crate) fn parse_records(blob: &[u8]) -> Result<(HashMap<&str, Array<'_>>, u32), ModelError> {
    let mut map = HashMap::new();
    let mut off = 0usize;
    let mut records = 0u32;
    while off < blob.len() {
        if off + RECORD_HEADER_SIZE > blob.len() {
            return Err(ModelError::Malformed);
        }
        let head = &blob[off..off + RECORD_HEADER_SIZE];
        if &head[0..4] != b"DNNw" {
            return Err(ModelError::Malformed);
        }
        // Upstream writes version 0 in every header. It used to be read and ignored,
        // which is how a blob from a newer format loads as a plausible-looking model.
        let version = i32::from_le_bytes([head[4], head[5], head[6], head[7]]);
        if version != RECORD_VERSION {
            return Err(ModelError::UnsupportedVersion(version as u32));
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

        let data_start = off + RECORD_HEADER_SIZE;
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
        records += 1;
        off = data_start + block_size;
    }
    Ok((map, records))
}

/// Look up one array and check its record type.
fn array<'a>(
    arrays: &HashMap<&str, Array<'a>>,
    name: &str,
    expected: i32,
) -> Result<&'a [u8], ModelError> {
    let a = arrays
        .get(name)
        .ok_or_else(|| ModelError::MissingArray(name.to_string()))?;
    if a.kind != expected {
        return Err(ModelError::BadType(name.to_string()));
    }
    Ok(a.data)
}

fn f32_array(arrays: &HashMap<&str, Array<'_>>, name: &str) -> Result<Vec<f32>, ModelError> {
    let data = array(arrays, name, WEIGHT_TYPE_FLOAT)?;
    if data.len() % 4 != 0 {
        return Err(ModelError::BadSize(name.to_string()));
    }
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn i32_array(arrays: &HashMap<&str, Array<'_>>, name: &str) -> Result<Vec<i32>, ModelError> {
    let data = array(arrays, name, WEIGHT_TYPE_INT32)?;
    if data.len() % 4 != 0 {
        return Err(ModelError::BadSize(name.to_string()));
    }
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn i8_array(arrays: &HashMap<&str, Array<'_>>, name: &str) -> Result<Vec<i8>, ModelError> {
    Ok(array(arrays, name, WEIGHT_TYPE_INT8)?
        .iter()
        .map(|&b| b as i8)
        .collect())
}

/// An optional array, validated against the layer it belongs to.
fn optional_f32(
    arrays: &HashMap<&str, Array<'_>>,
    name: &str,
    nb_outputs: usize,
) -> Result<Option<Vec<f32>>, ModelError> {
    if !arrays.contains_key(name) {
        return Ok(None);
    }
    let v = f32_array(arrays, name)?;
    if v.len() != nb_outputs {
        return Err(ModelError::BadSize(name.to_string()));
    }
    Ok(Some(v))
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
pub(crate) fn block_major_form(dense: &[f32], m: usize, n: usize) -> Vec<f32> {
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

/// Build one layer from the records named after its base (`linear_init`).
///
/// Every array name derives from the layer's base name, which is what lets the loader
/// accept either representation of the same layer: `<base>_weights_float` (the bit-exact
/// path) or `<base>_weights_int8` with the per-output `<base>_scale` (the quantized path).
/// `<base>_bias` is always required; `<base>_weights_idx` and `<base>_weights_diag` are
/// optional.
pub(crate) fn linear(
    arrays: &HashMap<&str, Array<'_>>,
    base: &str,
    nb_inputs: usize,
    nb_outputs: usize,
) -> Result<LinearLayer, ModelError> {
    let bias_name = format!("{base}_bias");
    let bias = f32_array(arrays, &bias_name)?;
    if bias.len() != nb_outputs {
        return Err(ModelError::BadSize(bias_name));
    }
    let diag_name = format!("{base}_weights_diag");
    let diag = optional_f32(arrays, &diag_name, nb_outputs)?;

    let int8_name = format!("{base}_weights_int8");
    if arrays.contains_key(int8_name.as_str()) {
        let weights = i8_array(arrays, &int8_name)?;
        if weights.len() != nb_inputs * nb_outputs {
            return Err(ModelError::BadSize(int8_name));
        }
        let scale_name = format!("{base}_scale");
        let scale = f32_array(arrays, &scale_name)?;
        if scale.len() != nb_outputs {
            return Err(ModelError::BadSize(scale_name));
        }
        // `corr` is derived data, not part of the blob format: it is the per-row `128 · Σ w` the
        // VNNI tier precomputes, so a loaded int8 model and a runtime-quantized one are the
        // same model.
        let corr: Vec<i32> = (0..nb_outputs)
            .map(|o| {
                128 * weights[o * nb_inputs..(o + 1) * nb_inputs]
                    .iter()
                    .map(|&v| v as i32)
                    .sum::<i32>()
            })
            .collect();
        return Ok(LinearLayer {
            bias,
            // The int8 form is always the transposed dense `[output][input]` matrix, so a
            // stored int8 blob reads straight into it with no permutation.
            weights: Weights::Q8 {
                w: weights,
                scale,
                corr,
            },
            weights_idx: None,
            diag,
            nb_inputs,
            nb_outputs,
            dense_layout: DenseLayout::RowMajor,
        });
    }

    let weights_name = format!("{base}_weights_float");
    let float_weights = f32_array(arrays, &weights_name)?;
    let idx_name = format!("{base}_weights_idx");
    let idx = if arrays.contains_key(idx_name.as_str()) {
        let idx = i32_array(arrays, &idx_name)?;
        let total_blocks = sparse_total_blocks(&idx, nb_outputs)
            .ok_or_else(|| ModelError::BadSize(weights_name.clone()))?;
        if float_weights.len() != 32 * total_blocks {
            return Err(ModelError::BadSize(weights_name));
        }
        Some(idx)
    } else {
        if float_weights.len() != nb_inputs * nb_outputs {
            return Err(ModelError::BadSize(weights_name));
        }
        None
    };

    // A complete, ascending sparse index is dense-equivalent: same values, same
    // accumulation order per output, so the fast GEMV tier can be used.
    let dense = idx
        .as_ref()
        .and_then(|idx| complete_ascending_dense_form(idx, &float_weights, nb_inputs, nb_outputs));
    let is_dense = dense.is_some() || idx.is_none();
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
        bias,
        weights: Weights::Float(stored),
        weights_idx: if is_dense { None } else { idx },
        diag,
        nb_inputs,
        nb_outputs,
        dense_layout: if use_blocks {
            DenseLayout::Blocks8
        } else {
            DenseLayout::RowMajor
        },
    })
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
        // LCG values are never NaN, so the ordering is total here.
        a.sort_by(|x, y| x.partial_cmp(y).expect("fixture values are never NaN"));
        b.sort_by(|x, y| x.partial_cmp(y).expect("fixture values are never NaN"));
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
    fn a_record_from_a_newer_format_is_rejected_by_version() {
        let mut record = vec![0u8; 64 + 4];
        record[..4].copy_from_slice(b"DNNw");
        record[4..8].copy_from_slice(&1i32.to_le_bytes()); // version, not 0
        record[8..12].copy_from_slice(&0i32.to_le_bytes()); // type float
        record[12..16].copy_from_slice(&4i32.to_le_bytes()); // size
        record[16..20].copy_from_slice(&4i32.to_le_bytes()); // block size
        record[20..30].copy_from_slice(b"conv1_bias");
        record[64..].copy_from_slice(&1.0f32.to_le_bytes());
        assert_eq!(
            parse_records(&record).err(),
            Some(ModelError::UnsupportedVersion(1))
        );
    }

    #[test]
    fn an_array_of_the_wrong_record_type_is_named() {
        let mut record = vec![0u8; 64 + 4];
        record[..4].copy_from_slice(b"DNNw");
        record[8..12].copy_from_slice(&1i32.to_le_bytes()); // int32, not float
        record[12..16].copy_from_slice(&4i32.to_le_bytes());
        record[16..20].copy_from_slice(&4i32.to_le_bytes());
        record[20..30].copy_from_slice(b"conv1_bias");
        record[64..].copy_from_slice(&1.0f32.to_le_bytes());
        let (arrays, records) = parse_records(&record).expect("structurally valid");
        assert_eq!(records, 1);
        assert_eq!(
            f32_array(&arrays, "conv1_bias").err(),
            Some(ModelError::BadType("conv1_bias".to_string()))
        );
    }

    #[test]
    fn int8_records_rebuild_the_derived_correction() {
        // `corr` is not in the blob format, so the loader must derive it exactly as
        // `quantize()` computes it: per output row, `128 · Σ w`. Two rows here, so the
        // expected values are checkable by hand: 128·(1−2+3) = 256 and 128·(4+5+6) = 1920.
        fn le_i8(values: &[i8]) -> Vec<u8> {
            values.iter().map(|&v| v as u8).collect()
        }
        fn le_f32(values: &[f32]) -> Vec<u8> {
            values.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        let w = le_i8(&[1, -2, 3, 4, 5, 6]);
        let scale = le_f32(&[0.5, 0.25]);
        let bias = le_f32(&[0.1, 0.2]);
        let arrays = HashMap::from([
            (
                "x_weights_int8",
                Array {
                    kind: WEIGHT_TYPE_INT8,
                    data: &w,
                },
            ),
            (
                "x_scale",
                Array {
                    kind: WEIGHT_TYPE_FLOAT,
                    data: &scale,
                },
            ),
            (
                "x_bias",
                Array {
                    kind: WEIGHT_TYPE_FLOAT,
                    data: &bias,
                },
            ),
        ]);
        let layer = linear(&arrays, "x", 3, 2).expect("a complete int8 layer");
        match &layer.weights {
            Weights::Q8 { corr, .. } => assert_eq!(corr, &[256, 1920][..]),
            Weights::Float(_) => assert!(
                matches!(layer.weights, Weights::Q8 { .. }),
                "int8 records must produce a Q8 layer"
            ),
        }
    }
}
