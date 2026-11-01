# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking changes

- The minimum supported Rust version is now **1.89** (was 1.85): the AVX-512 VNNI tier uses
  AVX-512 intrinsics and the `avx512f` target feature, which are stable from 1.89. AVX-512 is
  still only *used* when CPUID reports it.
- `ModelError` is `#[non_exhaustive]`, so an exhaustive `match` on it needs a wildcard arm.
  This lets later releases add error variants without another break.

### Added

- Repository scaffolding aligned with the sibling `lovelace-labs/*-rs` repos:
  `rust-toolchain.toml`, `rustfmt.toml`, `Makefile`, `VERSION`, `AUTHORS`,
  `.editorconfig`, `.gitattributes`, `.gitconfig`, `.gitmodules`,
  `.dockerignore`, `cspell.json`, `codemeta.json`, `CITATION.cff`,
  `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, and `AGENTS.md`.
- `.github/workflows/ci.yml` aligned with the org-wide CI template.
- Runtime-dispatched x86-64 SIMD tiers, chosen from CPUID with no `-C target-cpu` needed:
  int8 dot products (SSE2, SSE4.1, AVX2, AVX-512 VNNI), the float GEMVs (including a
  register-resident accumulator for the sparse 8x4 GEMV and a block-major dense layout),
  the pitch cross-correlation and the FFT butterflies. Every tier keeps the reference
  accumulation order and is asserted bit-identical to the portable scalar code
  (`src/simd/tests.rs`, `src/fft/tests.rs`). On an AMD Zen 3 core the int8 network drops
  from ≈487 µs to ≈84 µs per frame and the float full frame from ≈271 µs to ≈210 µs.
- A sparse weight index that is complete and ascending (the shipped GRU matrices) is loaded
  in dense form at parse time; the accumulation order, and therefore the output, is
  unchanged.

### Changed

- Constructors follow the `new_with_foo()` form: `DenoiseState::new_with_model()` and
  `DenoiseStateV1::new_with_model()`. The old `with_model` names remain as deprecated
  aliases.
- The model-geometry constants live in a `params` module and are re-exported from the crate
  root, so `rnnoise::FRAME_SIZE` and friends keep their paths. No `unwrap()` is left in
  `src/`. `tests/api_surface.rs` names every exported item, so a public rename is a compile
  failure.

## [0.1.0]

- Initial release.
