//! x86-64 kernels — int8 dot products, the float GEMV and the pitch cross-correlation,
//! one file per ISA tier. The tier is chosen at runtime
//! by [`crate::simd::dispatch`], never by a build-wide `-C target-cpu` setting:
//! the crate is compiled for the plain `x86-64` baseline, and `#[target_feature]`
//! is applied only to the kernel functions themselves.
//!
//! ```text
//! v1  SSE2    punpcklbw + cmpgt sign mask + pmaddwd   (no byte-widen instruction)
//! v2  SSSE3   pmovsxbw + pmaddwd                      (8 lanes)
//! v3  AVX2    pmovsxbw + pmaddwd, 16 lanes            (32 MACs per instruction pair)
//! v4  VNNI    vpdpbusd, 16 lanes                      (64 MACs per instruction)
//! ```
//!
//! The float kernels (`dense`, `sparse8x4`, `xcorr`) use the SSE2 baseline tier and an
//! AVX2 tier where it helps.
//!
//! Tier v4 needs Rust ≥ 1.89 (the AVX-512 intrinsics and the `avx512f` target
//! feature are stable from there), which is why `rust-version` is 1.89. It is still
//! only *used* when CPUID reports `avx512f` + `avx512vnni`, so a Rust 1.89+ binary
//! built on an AVX2-only machine runs tier v3.

pub(crate) mod avx2;
pub(crate) mod dense;
pub(crate) mod dense_blocks;
pub(crate) mod select;
pub(crate) mod sparse8x4;
pub(crate) mod sse2;
pub(crate) mod sse41;
pub(crate) mod vnni;
pub(crate) mod xcorr;
