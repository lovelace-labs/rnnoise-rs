//! Tier v4 — AVX-512 VNNI (`vpdpbusd`): four int8 MACs per lane in one instruction, i.e.
//! 64 MACs per iteration — the only tier that does real int8 multiply-accumulate in hardware.
//!
//! `vpdpbusd` is **unsigned × signed**, so one side has to live in the biased `u8` domain. The
//! `+128` is paid once **per layer**, not once per chunk: the activations are biased at the top
//! of the call, and each row subtracts its own `128 · Σ w`, which `Weights::Q8` precomputes at
//! quantize time. The identity
//!
//! ```text
//! Σ u8(q_x + 128) · s8(w) − 128 · Σ s8(w)  ==  Σ s8(q_x) · s8(w)
//! ```
//!
//! is exact in `i32` (products ≤ `255 · 127`, a row sum ≤ `1536 · 32385`), so the tier stays
//! bit-identical to `simd::scalar` and the parity numbers do not move.
//!
//! Every quantized layer in the model is 384 bytes wide = exactly 6 chunks, so the shape that
//! actually runs is the 6-chunk one, where the biased activations live in registers for the whole
//! row loop: a row is 6 weight loads + 6 `vpdpbusd` + one combine + one reduce. Measured per row
//! on a Zen 4 core (Ryzen 7 7800X3D), same binary, same data, isolated kernel, `m = 384`,
//! `n = 1152`:
//!
//! | shape | ns per row |
//! |---|--:|
//! | `vpaddb` in the loop, one accumulator (the shape before this change) | 4.88 |
//! | bias hoisted into a buffer, one accumulator | 4.77 |
//! | bias hoisted into registers, four accumulators (this tier) | **3.06** |
//!
//! Four accumulators *on their own* measured worse (5.44 ns): at 6 chunks per row this tier is
//! load-port bound, not dependency-chain bound, so the four zeroings plus the extra combine cost
//! more than the latency they hide. Keeping the activations in registers is what pays — it
//! removes 6 double-pumped load µops per row, which is the difference between 4.77 and 3.06.
//!
//! Needs Rust ≥ 1.89 for the AVX-512 intrinsics and the `avx512f` target feature (hence the
//! crate's `rust-version`), and the CPU must report `avx512f` + `avx512bw` + `avx512vnni`:
//! `vpdpbusd` is VNNI, the `+128` bias uses AVX512BW `add_epi8` (`x + 128 mod 256` is also
//! `x XOR 0x80`, which is what LLVM emits here). Nothing forces AVX-512 codegen anywhere else:
//! `dispatch` calls in only when CPUID reports the CPU has it.

use core::arch::x86_64::*;

use crate::params::MAX_Q8_INPUTS;

#[inline(always)]
unsafe fn hsum16(acc: __m512i) -> i32 {
    _mm512_reduce_add_epi32(acc)
}

#[inline(always)]
unsafe fn combine4(a0: __m512i, a1: __m512i, a2: __m512i, a3: __m512i) -> __m512i {
    _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3))
}

// SAFETY: requires AVX-512F + AVX512BW + AVX-512 VNNI, all three runtime-checked in `select`.
// `m == MAX_Q8_INPUTS` is checked by the caller, so the six loads below read exactly the six
// chunks the reference zips to; `corr[o]` is the `128 · Σ w` correction for that row, so the row
// result is the reference's integer sum. Four independent accumulator chains, because a single
// `vpdpbusd` chain is what made the old shape slower than tier v3.
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
#[target_feature(enable = "avx512vnni")]
unsafe fn dense_q8_hot(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    corr: &[i32],
    n: usize,
    qx: &[i8],
) {
    let sign = _mm512_set1_epi8(-128); // 0x80: the unsigned domain
    let a = qx.as_ptr() as *const __m512i;
    let x0 = _mm512_add_epi8(_mm512_loadu_si512(a), sign);
    let x1 = _mm512_add_epi8(_mm512_loadu_si512(a.add(1)), sign);
    let x2 = _mm512_add_epi8(_mm512_loadu_si512(a.add(2)), sign);
    let x3 = _mm512_add_epi8(_mm512_loadu_si512(a.add(3)), sign);
    let x4 = _mm512_add_epi8(_mm512_loadu_si512(a.add(4)), sign);
    let x5 = _mm512_add_epi8(_mm512_loadu_si512(a.add(5)), sign);
    for o in 0..n {
        let b = w.as_ptr().add(o * MAX_Q8_INPUTS) as *const __m512i;
        let z = _mm512_setzero_si512();
        // Four independent chains over six chunks: `a0` and `a1` take two chunks each.
        let mut a0 = _mm512_dpbusd_epi32(z, x0, _mm512_loadu_si512(b));
        let mut a1 = _mm512_dpbusd_epi32(z, x1, _mm512_loadu_si512(b.add(1)));
        let a2 = _mm512_dpbusd_epi32(z, x2, _mm512_loadu_si512(b.add(2)));
        let a3 = _mm512_dpbusd_epi32(z, x3, _mm512_loadu_si512(b.add(3)));
        a0 = _mm512_dpbusd_epi32(a0, x4, _mm512_loadu_si512(b.add(4)));
        a1 = _mm512_dpbusd_epi32(a1, x5, _mm512_loadu_si512(b.add(5)));
        out[o] = (hsum16(combine4(a0, a1, a2, a3)) - corr[o]) as f32 * scale[o];
    }
}

// SAFETY: same three feature checks as above, done by `select`. Unaligned 64-byte loads over
// `min(w.len(), x.len())` — the length the reference zips to — plus the byte tail, so nothing
// reads past either slice. `len` is at most `MAX_Q8_INPUTS` because that is the width of the
// activation buffer `nnet` quantizes into, so the scratch buffer here is exactly one row.
// This is the general form for widths that are not the hot width; the model never runs here (every
// quantized layer is 384 bytes), but a custom model and the tail tests do.
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
#[target_feature(enable = "avx512vnni")]
unsafe fn dense_q8_general(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    corr: &[i32],
    m: usize,
    n: usize,
    qx: &[i8],
) {
    let len = m.min(qx.len());
    let chunks = len / 64;
    let sign = _mm512_set1_epi8(-128);
    let mut xu = [0u8; MAX_Q8_INPUTS];
    for k in 0..chunks {
        let v = _mm512_add_epi8(
            _mm512_loadu_si512(qx.as_ptr().add(k * 64) as *const __m512i),
            sign,
        );
        _mm512_storeu_si512(xu.as_mut_ptr().add(k * 64) as *mut __m512i, v);
    }
    for k in chunks * 64..len {
        xu[k] = (qx[k] as i32 + 128) as u8;
    }
    for o in 0..n {
        let mut acc = _mm512_setzero_si512();
        for k in 0..chunks {
            acc = _mm512_dpbusd_epi32(
                acc,
                _mm512_loadu_si512(xu.as_ptr().add(k * 64) as *const __m512i),
                _mm512_loadu_si512(w.as_ptr().add(o * m + k * 64) as *const __m512i),
            );
        }
        // `corr[o]` covers the whole row, tail included, so the tail can stay in the
        // unsigned × signed form.
        let mut sum = hsum16(acc) - corr[o];
        for k in chunks * 64..len {
            sum += xu[k] as i32 * w[o * m + k] as i32;
        }
        out[o] = sum as f32 * scale[o];
    }
}

pub(crate) unsafe fn dense_q8(
    out: &mut [f32],
    w: &[i8],
    scale: &[f32],
    corr: &[i32],
    m: usize,
    n: usize,
    qx: &[i8],
) {
    if m == MAX_Q8_INPUTS && qx.len() >= m {
        dense_q8_hot(out, w, scale, corr, n, qx);
        return;
    }
    dense_q8_general(out, w, scale, corr, m, n, qx);
}
