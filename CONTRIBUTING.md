# Contributing to rnnoise-rs

Thank you for your interest in contributing! This document describes the
workflow used across all `lovelace-labs/*-rs` repos.

## Getting started

1. **Fork** the repository on GitHub.
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rnnoise-rs.git
   cd rnnoise-rs
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/lovelace-labs/rnnoise-rs.git
   ```

## Development setup

### Prerequisites

- Rust 1.89+ (edition 2021), installed via [rustup](https://rustup.rs/).
  The pinned toolchain plus `rustfmt` and `clippy` are declared in
  [`rust-toolchain.toml`](rust-toolchain.toml) and installed automatically.

### Building

```bash
make build       # cargo build
make test        # cargo test
make check       # fmt-check + clippy + spell + test + deny + rt-gate (the CI gate)
```

Run `make help` for the full list of targets.

## Branch naming

Use descriptive branch names with a prefix:

- `feat/` — new features
- `fix/` — bug fixes
- `docs/` — documentation changes
- `refactor/` — code refactoring
- `test/` — test additions or changes
- `chore/` — maintenance tasks

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

## Pull request process

1. Sync with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/dev
   ```
2. Run `make check`.
3. Open a pull request with a clear title (Conventional Commits format) and
   a description of what changed and why. Link related issues.
4. Address review feedback by pushing additional commits.
5. Squash and merge after approval.

## CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs five jobs:

| job | what it checks |
|---|---|
| `lint` | `cargo fmt --check`, clippy `-D warnings`, `cargo deny check`, and cspell |
| `test` | the suite on ubuntu x86-64, ubuntu-arm aarch64, macOS (Apple Silicon) and Windows, debug and release |
| `simd` | each kernel path, forced with `RNNOISE_SIMD_TIER`, over the tiers the runner has |
| `c-reference` | the pinned scalar C reference built in CI and compared with the committed goldens |
| `bench` | criterion per architecture **and per tier** (`simd-tier-override` feature + `RNNOISE_SIMD_TIER`), JSON results uploaded as artifacts |

`simd` requests only tiers its runners actually have: requesting a tier the CPU lacks is a
failure, not a skip, because CI promised that coverage. `c-reference` and the parity test
skip gracefully when the toolchain, the network or the clips are absent.

## Coding standards

- Run `cargo fmt` (enforced by CI).
- All clippy warnings are errors (`-D warnings`).
- Document public items with `///` comments.
- Prefer `Result` over panics for recoverable errors; `Option` for obvious
  failures; `panic!` only for violated invariants.
- No `unwrap` in `src/` — pattern-match when the input proves the case, otherwise
  `expect("context")`. Tests use `expect` with an assertion context string.
- Unsafe only in the declared islands (`src/capi.rs`, `src/simd/`,
  `src/fft/x86_64.rs`, `src/fft/dispatch.rs`), each with a `SAFETY:` justification.
- No `catch_unwind`, no threads, no unstable features, no async in the DSP path, nothing
  before `main()`, no `rustfmt::skip` except manual alignment.
- The gate is `make check`: `fmt-check`, `clippy`, `spell`, `test`, `deny`,
  `rt-gate`. `spell` runs cspell; `rt-gate`
  (`scripts/realtime_gate.py`) checks the real-time contract per function — no allocation,
  lock, lazy static, I/O, logging or CPUID probe inside a per-frame function — and
  `tests/real_time_no_alloc.rs` proves it at runtime.

## Module organization

- `lib.rs` and `mod.rs` only declare submodules and re-export the public
  surface — no logic or types in module roots.
- Prefer small, single-responsibility submodules over large catch-all files.

## Questions?

- [Issues](https://github.com/lovelace-labs/rnnoise-rs/issues) — bugs and feature
  requests.
- [Discussions](https://github.com/lovelace-labs/rnnoise-rs/discussions) — design
  questions.
- [SECURITY.md](SECURITY.md) — vulnerability reporting.
