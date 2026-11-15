//! Tier-vs-reference equality. A tier whose CPU feature is not present is skipped,
//! so these run anywhere (and in CI on whatever the runner happens to be).

// SAFETY: these tests call each tier directly (no runtime detection) to prove it is
// bit-identical to the portable reference; the target_feature guard is the caller's.

use crate::simd::scalar;
use crate::simd::Tier;

const M: usize = 384; // a real layer width (conv2 / GRU)
const ROWS: usize = 64;

#[cfg(target_arch = "x86_64")]
const LADDER: &[Tier] = &[
    Tier::Vnni,
    Tier::Avx2,
    Tier::Sse41,
    Tier::Sse2,
    Tier::Scalar,
];
#[cfg(target_arch = "aarch64")]
const LADDER: &[Tier] = &[Tier::Dotprod, Tier::Scalar];
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const LADDER: &[Tier] = &[Tier::Scalar];

#[cfg(target_arch = "x86_64")]
const TIER_NAMES: &[Tier] = &[
    Tier::Scalar,
    Tier::Sse2,
    Tier::Sse41,
    Tier::Avx2,
    Tier::Vnni,
];
#[cfg(target_arch = "aarch64")]
const TIER_NAMES: &[Tier] = &[Tier::Scalar, Tier::Dotprod];
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const TIER_NAMES: &[Tier] = &[Tier::Scalar];

/// `128 · Σ_q w` per row — exactly what `LinearLayer::quantize` stores for tier v4.
fn row_corr(w: &[i8], m: usize, n: usize) -> Vec<i32> {
    (0..n)
        .map(|o| 128 * w[o * m..o * m + m].iter().map(|&v| v as i32).sum::<i32>())
        .collect()
}

fn fixture() -> (Vec<i8>, Vec<i8>, Vec<f32>, Vec<i32>) {
    let mut s = 7u32;
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s >> 24) as i32).clamp(-127, 127) as i8
    };
    let w: Vec<i8> = (0..ROWS * M).map(|_| next()).collect();
    let qx: Vec<i8> = (0..M).map(|_| next()).collect();
    let scale: Vec<f32> = (0..ROWS).map(|i| (i as f32 + 1.0) * 1e-4).collect();
    let corr = row_corr(&w, M, ROWS);
    (w, qx, scale, corr)
}

/// Reports what this CPU can actually run. A green CI run is not coverage of a tier that
/// was skipped, so the test says which tiers it exercised; `is_x86_feature_detected!` needs
/// a literal feature name, so there is no loop here.
#[cfg(target_arch = "x86_64")]
fn tier_coverage(skipped: &[&str]) {
    if skipped.is_empty() {
        eprintln!("x86-64 tier coverage: every tier ran here");
    } else {
        eprintln!("x86-64 tier coverage: skipped {skipped:?} (not detected on this CPU)");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn every_x86_tier_matches_scalar() {
    let (w, qx, scale, corr) = fixture();
    let mut reference = vec![0.0f32; ROWS];
    scalar::dense_q8(&mut reference, &w, &scale, &corr, M, ROWS, &qx);

    let mut out = vec![0.0f32; ROWS];
    unsafe { crate::simd::x86_64::sse2::dense_q8(&mut out, &w, &scale, &corr, M, ROWS, &qx) };
    assert_eq!(out, reference, "tier v1 (SSE2) diverged");

    let mut skipped = Vec::new();
    let sse41 = std::arch::is_x86_feature_detected!("sse4.1");
    let avx2 = std::arch::is_x86_feature_detected!("avx2");
    let vnni = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vnni");

    if sse41 {
        out.iter_mut().for_each(|v| *v = 0.0);
        unsafe { crate::simd::x86_64::sse41::dense_q8(&mut out, &w, &scale, &corr, M, ROWS, &qx) };
        assert_eq!(out, reference, "tier v2 (SSE4.1) diverged");
    } else {
        skipped.push("v2 (SSE4.1)");
    }
    if avx2 {
        out.iter_mut().for_each(|v| *v = 0.0);
        unsafe { crate::simd::x86_64::avx2::dense_q8(&mut out, &w, &scale, &corr, M, ROWS, &qx) };
        assert_eq!(out, reference, "tier v3 (AVX2) diverged");
    } else {
        skipped.push("v3 (AVX2)");
    }
    if vnni {
        // The hot width: tier v4's 6-chunk path, activations in registers.
        out.iter_mut().for_each(|v| *v = 0.0);
        unsafe { crate::simd::x86_64::vnni::dense_q8(&mut out, &w, &scale, &corr, M, ROWS, &qx) };
        assert_eq!(out, reference, "tier v4 (VNNI) diverged");
    } else {
        skipped.push("v4 (VNNI)");
    }

    // Widths that are not a multiple of the vector size exercise the tail loops, and a row
    // narrower than the activation buffer needs the narrower `128 · Σ w` correction.
    for width in [7usize, 33, 65] {
        let c = row_corr(&w, width, ROWS);
        let mut r = vec![0.0f32; ROWS];
        let mut o = vec![0.0f32; ROWS];
        scalar::dense_q8(&mut r, &w, &scale, &c, width, ROWS, &qx);
        unsafe { crate::simd::x86_64::sse2::dense_q8(&mut o, &w, &scale, &c, width, ROWS, &qx) };
        assert_eq!(o, r, "tier v1 tail, width {width}");
        unsafe { crate::simd::x86_64::avx2::dense_q8(&mut o, &w, &scale, &c, width, ROWS, &qx) };
        assert_eq!(o, r, "tier v3 tail, width {width}");
        if vnni {
            unsafe {
                crate::simd::x86_64::vnni::dense_q8(&mut o, &w, &scale, &c, width, ROWS, &qx)
            };
            assert_eq!(o, r, "tier v4 tail, width {width}");
        }
    }

    tier_coverage(&skipped);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn sdot_matches_scalar() {
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return;
    }
    let (w, qx, scale, corr) = fixture();
    let mut reference = vec![0.0f32; ROWS];
    scalar::dense_q8(&mut reference, &w, &scale, &corr, M, ROWS, &qx);
    let mut out = vec![0.0f32; ROWS];
    unsafe { crate::simd::aarch64::dotprod::dense_q8(&mut out, &w, &scale, &corr, M, ROWS, &qx) };
    assert_eq!(out, reference, "sdot diverged");
}

#[test]
fn vnni_bias_identity_is_exact() {
    // `vpdpbusd` is unsigned × signed, so tier v4 accumulates `Σ u8(q_x + 128) · w` in the hot
    // loop and subtracts `128 · Σ w` per row (the `row_corr` table). That is the whole
    // correctness argument for the tier, and it is checkable without an AVX-512 CPU — CI runners
    // usually have none.
    let (w, qx, _, _) = fixture();
    for row_len in [7usize, 33, 65, M] {
        let row = &w[..row_len];
        let plain: i32 = row
            .iter()
            .zip(&qx)
            .map(|(&a, &b)| a as i32 * b as i32)
            .sum();
        let biased: i32 = row
            .iter()
            .zip(&qx)
            .map(|(&a, &b)| a as i32 * (b as i32 + 128))
            .sum();
        let sum_w: i32 = row.iter().map(|&v| v as i32).sum();
        assert_eq!(
            biased - 128 * sum_w,
            plain,
            "tier v4 bias correction is not exact at row length {row_len}"
        );
        assert!(
            biased.abs() <= 384 * 255 * 127,
            "i32 accumulator would overflow"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_dense_sgemv_matches_scalar() {
    // Model-shaped widths (conv2, the GRUs) plus a width that is not a multiple of the
    // vector size, which exercises the per-output tail.
    let mut s = 21u32;
    let mut rng = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s >> 8) as f32) / 16_777_216.0 - 0.5
    };
    for (m, n) in [(384usize, 1152), (384, 384), (1536, 32), (65, 128), (7, 33)] {
        let w: Vec<f32> = (0..m * n).map(|_| rng()).collect();
        let x: Vec<f32> = (0..m).map(|_| rng()).collect();
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        scalar::dense_sgemv(&mut a, &w, m, n, &x);
        unsafe { crate::simd::x86_64::dense::dense_sgemv(&mut b, &w, m, n, &x) };
        assert_eq!(b, a, "AVX2 dense GEMV diverged for {m}x{n}");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn xcorr_tiers_match_scalar() {
    // Pitch-search shape: `y` holds `n + max_pitch` entries. Sizes that are not a
    // multiple of the vector width exercise the per-lag tail of every tier.
    let n = 240;
    for max_pitch in [147usize, 384, 5] {
        let mut s = 99u32;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / 16_777_216.0 - 0.5
        };
        let x: Vec<f32> = (0..n).map(|_| next()).collect();
        let y: Vec<f32> = (0..n + max_pitch).map(|_| next()).collect();
        let mut reference = vec![0.0f32; max_pitch];
        scalar::xcorr(&mut reference, &x, &y, n, max_pitch);

        let mut out = vec![0.0f32; max_pitch];
        unsafe { crate::simd::x86_64::xcorr::xcorr4(&mut out, &x, &y, n, max_pitch) };
        assert_eq!(
            out, reference,
            "xcorr tier v1 (SSE2) diverged at {max_pitch}"
        );
        if std::arch::is_x86_feature_detected!("avx2") {
            out.iter_mut().for_each(|v| *v = 0.0);
            unsafe { crate::simd::x86_64::xcorr::xcorr8(&mut out, &x, &y, n, max_pitch) };
            assert_eq!(
                out, reference,
                "xcorr tier v3 (AVX2) diverged at {max_pitch}"
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_sparse8x4_matches_scalar() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    let (rows, cols_per_group) = (64usize, 9usize);
    let mut s = 11u32;
    let mut rng = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s >> 8) as f32) / 16_777_216.0 - 0.5
    };
    let w: Vec<f32> = (0..(rows / 8) * cols_per_group * 32)
        .map(|_| rng())
        .collect();
    let x: Vec<f32> = (0..64).map(|_| rng()).collect();
    let mut idx: Vec<i32> = Vec::new();
    for g in 0..rows / 8 {
        idx.push(cols_per_group as i32);
        for c in 0..cols_per_group {
            idx.push(((g * 4 + c) % 64) as i32);
        }
    }
    let mut a = vec![0.0f32; rows];
    let mut b = vec![0.0f32; rows];
    scalar::sparse_sgemv8x4(&mut a, &w, &idx, rows, &x);
    unsafe { crate::simd::x86_64::sparse8x4::sparse_sgemv8x4(&mut b, &w, &idx, rows, &x) };
    assert_eq!(
        b, a,
        "AVX2 sparse kernel diverged from the scalar reference"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn block_major_tier_matches_row_major_reference() {
    // The block form is the same values permuted (`w[(o/8)*8*m + j*8 + o%8]`), and each
    // output still accumulates in ascending `j`, so the bits must be identical. The sizes
    // include a non-multiple-of-8 `n`, which exercises the per-output tail.
    let mut s = 31u32;
    let mut rng = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s >> 8) as f32) / 16_777_216.0 - 0.5
    };

    fn block_form(row: &[f32], m: usize, n: usize) -> Vec<f32> {
        let groups = n / 8;
        let mut out = vec![0.0f32; row.len()];
        for j in 0..m {
            for o in 0..n {
                let dst = if o < groups * 8 {
                    (o / 8) * 8 * m + j * 8 + (o % 8)
                } else {
                    groups * 8 * m + (o - groups * 8) * m + j
                };
                out[dst] = row[j * n + o];
            }
        }
        out
    }

    for (m, n) in [
        (384usize, 1152),
        (384, 384),
        (1536, 32),
        (65, 128),
        (7, 33),
        (12, 8),
    ] {
        let row: Vec<f32> = (0..m * n).map(|_| rng()).collect();
        let x: Vec<f32> = (0..m).map(|_| rng()).collect();
        let block = block_form(&row, m, n);

        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        scalar::dense_sgemv(&mut a, &row, m, n, &x);
        scalar::dense_sgemv_blocks(&mut b, &block, m, n, &x);
        assert_eq!(b, a, "block-major reference diverged for {m}x{n}");

        if std::arch::is_x86_feature_detected!("avx2") {
            b.iter_mut().for_each(|v| *v = 0.0);
            unsafe {
                crate::simd::x86_64::dense_blocks::dense_sgemv_blocks(&mut b, &block, m, n, &x)
            };
            assert_eq!(b, a, "AVX2 block-major tier diverged for {m}x{n}");
        }
    }
}

/// The resolver is pure data (no `unsafe`, no kernel call): it must never hand back a
/// tier the CPU lacks, and a forced tier is capped to the best one a kernel actually
/// implements. `simd::tier` is the single decision point the CI job relies on.
#[test]
fn resolve_never_runs_a_tier_the_runner_lacks() {
    for &tier in TIER_NAMES {
        if !tier.is_available() {
            continue;
        }
        let guard = Tier::force(tier);
        let ran = Tier::resolve(LADDER);
        drop(guard);
        assert!(
            ran.is_available(),
            "resolve handed back a tier this CPU lacks"
        );
        assert!(
            LADDER.contains(&ran),
            "resolve returned a tier outside the ladder"
        );
        assert!(
            ran.rank() <= tier.rank(),
            "resolve ran a tier above the forced one"
        );
    }
}

/// A kernel with no v2 form must fall back to its v1 kernel when v2 is forced, and
/// forcing `scalar` must actually run the portable reference.
#[cfg(target_arch = "x86_64")]
#[test]
fn a_forced_tier_is_capped_to_the_kernel_that_has_it() {
    use crate::simd::x86_64::select::{AVX2_ONLY_LADDER, INT8_LADDER};

    let guard = Tier::force(Tier::Scalar);
    assert_eq!(Tier::resolve(INT8_LADDER), Tier::Scalar);
    assert_eq!(Tier::resolve(AVX2_ONLY_LADDER), Tier::Scalar);
    drop(guard);

    if Tier::Sse41.is_available() {
        let guard = Tier::force(Tier::Sse41);
        assert_eq!(Tier::resolve(INT8_LADDER), Tier::Sse41);
        assert_eq!(
            Tier::resolve(AVX2_ONLY_LADDER),
            Tier::Scalar,
            "no v2 kernel"
        );
        drop(guard);
    }
    if Tier::Avx2.is_available() {
        let guard = Tier::force(Tier::Avx2);
        assert_eq!(Tier::resolve(INT8_LADDER), Tier::Avx2);
        assert_eq!(Tier::resolve(AVX2_ONLY_LADDER), Tier::Avx2);
        drop(guard);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn a_forced_tier_is_capped_to_the_kernel_that_has_it() {
    use crate::simd::aarch64::select::LADDER;

    let guard = Tier::force(Tier::Scalar);
    assert_eq!(Tier::resolve(LADDER), Tier::Scalar);
    drop(guard);

    if Tier::Dotprod.is_available() {
        let guard = Tier::force(Tier::Dotprod);
        assert_eq!(Tier::resolve(LADDER), Tier::Dotprod);
        drop(guard);
    }
}

#[test]
fn tier_names_are_the_ci_interface() {
    // `RNNOISE_SIMD_TIER` is parsed from the same strings the coverage table prints.
    for &tier in TIER_NAMES {
        assert_eq!(Tier::parse(tier.name()), Some(tier), "name/parse mismatch");
    }
    assert_eq!(Tier::parse("nonsense"), None);
}
