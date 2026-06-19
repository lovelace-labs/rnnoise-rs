//! C ABI compatible with `rnnoise.h` (enable with the `capi` feature).
//!
//! The state and model are opaque, heap-allocated handles. Unlike the C
//! library, `rnnoise_get_size`/`rnnoise_init` (which assume a caller-allocated
//! struct of known size) are not meaningfully supported — use
//! `rnnoise_create`/`rnnoise_destroy` instead.

use std::os::raw::{c_char, c_float, c_int, c_void};
use std::sync::Arc;

use crate::{DenoiseState, RnnModel, FRAME_SIZE};

/// Opaque model handle (`RNNModel`).
pub struct RNNModel(Arc<RnnModel>);

/// Number of samples processed per `rnnoise_process_frame` call.
#[no_mangle]
pub extern "C" fn rnnoise_get_frame_size() -> c_int {
    FRAME_SIZE as c_int
}

/// Allocate and initialise a denoiser. `model` may be NULL for the default model.
///
/// # Safety
/// `model`, if non-NULL, must come from `rnnoise_model_from_buffer`.
#[no_mangle]
pub unsafe extern "C" fn rnnoise_create(model: *const RNNModel) -> *mut DenoiseState {
    let st = if model.is_null() {
        DenoiseState::new()
    } else {
        DenoiseState::with_model((*model).0.clone())
    };
    Box::into_raw(Box::new(st))
}

/// Free a denoiser created by `rnnoise_create`.
///
/// # Safety
/// `st` must be a pointer returned by `rnnoise_create` (or NULL).
#[no_mangle]
pub unsafe extern "C" fn rnnoise_destroy(st: *mut DenoiseState) {
    if !st.is_null() {
        drop(Box::from_raw(st));
    }
}

/// Denoise one frame; returns the VAD probability.
///
/// # Safety
/// `st` must be valid; `out` and `inp` must each point to at least
/// `rnnoise_get_frame_size()` floats.
#[no_mangle]
pub unsafe extern "C" fn rnnoise_process_frame(
    st: *mut DenoiseState,
    out: *mut c_float,
    inp: *const c_float,
) -> c_float {
    let st = &mut *st;
    let out = std::slice::from_raw_parts_mut(out, FRAME_SIZE);
    let inp = std::slice::from_raw_parts(inp, FRAME_SIZE);
    st.process_frame(out, inp)
}

/// Load a model from a memory buffer (`rnnoise_model_from_buffer`).
/// Returns NULL on failure.
///
/// # Safety
/// `ptr` must point to `len` valid bytes.
#[no_mangle]
pub unsafe extern "C" fn rnnoise_model_from_buffer(
    ptr: *const c_void,
    len: c_int,
) -> *mut RNNModel {
    if ptr.is_null() || len <= 0 {
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(ptr as *const u8, len as usize);
    match RnnModel::from_bytes(bytes) {
        Ok(m) => Box::into_raw(Box::new(RNNModel(Arc::new(m)))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Load a model from a file name (`rnnoise_model_from_filename`).
/// Returns NULL on failure.
///
/// # Safety
/// `filename` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn rnnoise_model_from_filename(filename: *const c_char) -> *mut RNNModel {
    if filename.is_null() {
        return std::ptr::null_mut();
    }
    let cstr = std::ffi::CStr::from_ptr(filename);
    let path = match cstr.to_str() {
        Ok(p) => p,
        Err(_) => return std::ptr::null_mut(),
    };
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return std::ptr::null_mut(),
    };
    match RnnModel::from_bytes(&bytes) {
        Ok(m) => Box::into_raw(Box::new(RNNModel(Arc::new(m)))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a model handle (`rnnoise_model_free`).
///
/// # Safety
/// `model` must come from a `rnnoise_model_from_*` call (or be NULL).
#[no_mangle]
pub unsafe extern "C" fn rnnoise_model_free(model: *mut RNNModel) {
    if !model.is_null() {
        drop(Box::from_raw(model));
    }
}
