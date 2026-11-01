//! Runtime-dispatched kernels: the int8 dot products for the quantized matmuls, the
//! float GEMV (row-major and block-major), and the pitch cross-correlation.
//!
//! Every int8 kernel computes the **same exact `i32` result**: for the model's widths
//! every product and pair sum fits in `i32` (max `384 · 127 · 127 ≈ 6.2e6`), so no
//! path can round differently and output never depends on which tier ran.
//!
//! The float kernels are exact in a different way: they keep the reference's
//! accumulation order (one accumulator per output element, or per lag, in `j` order),
//! and never fuse a multiply with an add, so a tier produces the same bits as
//! [`scalar`] on the same input.
//!
//! Which tier runs is decided at **runtime** by `dispatch` (CPUID + cached feature
//! flags), not at build time: the crate compiles for the plain `x86-64` baseline
//! and still uses AVX2/VNNI on a CPU that has them. `#[target_feature]` is only
//! used on the kernel functions themselves, never as a build-wide setting.

mod dispatch;
mod scalar;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

#[cfg(test)]
mod tests;

#[cfg(feature = "legacy-model")]
pub(crate) use dispatch::i8_dot;
pub(crate) use dispatch::{dense_q8, dense_sgemv, dense_sgemv_blocks, sparse_sgemv8x4, xcorr};
