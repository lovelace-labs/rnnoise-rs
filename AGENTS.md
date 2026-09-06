# AGENTS.md — rnnoise-rs

Instructions for AI agents working in this repository.

## Read first

1. [README.md](README.md) — repo overview and quick start.
2. [CONTRIBUTING.md](CONTRIBUTING.md) — branching, commit, and review rules.
3. The upstream reference: **RNNoise** (Xiph.Org / Jean-Marc Valin) — this crate is a faithful Rust port.

## What this repo is

Idiomatic Rust port of Xiph RNNoise (current conv+GRU model) for real-time speech noise suppression.

## Hard rules

1. **Faithful to upstream.** The default code path mirrors the upstream
   reference. Any change that alters numerical behaviour must be justified
   and validated against the parity harness — do not let parity regress.
2. **No new heavy dependencies** without cause; prefer pure-Rust crates and
   keep optional features opt-in.
3. **`#![forbid(unsafe_code)]`** stays where it is set; the crate aims to be
   unsafe-free.
4. **No silent breaking changes.** Public API changes need an entry in
   [CHANGELOG.md](CHANGELOG.md).

## Structure & quality conventions

Acceptance criteria for any new or refactored code (binding):

1. **`mod.rs` / `lib.rs` only re-export** — module roots declare submodules
   and re-export the public surface; no logic or types in roots.
2. **Highly modular** — small, single-responsibility submodules instead of
   large catch-all files.
3. **Consistent layout** — keep the module layout aligned with the sibling
   `lovelace-labs/*-rs` repos.
4. **Test coverage** — functionality ships with tests; tests needing weights
   or large data must skip gracefully when the artifacts are absent.

## Build & test

```sh
make check       # fmt + clippy + test (the CI gate)
```

Runs `cargo fmt --all --check`, `cargo clippy --all-targets -- -D warnings`,
and `cargo test`. See [Makefile](Makefile) for the full list of targets.

## Git

Commit as the repository's configured author. Never add AI / Claude
attribution, `Co-Authored-By`, or "Generated with" lines.
