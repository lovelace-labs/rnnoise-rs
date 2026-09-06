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

- Rust 1.85+ (edition 2021), installed via [rustup](https://rustup.rs/).
  The pinned toolchain plus `rustfmt` and `clippy` are declared in
  [`rust-toolchain.toml`](rust-toolchain.toml) and installed automatically.

### Building

```bash
make build       # cargo build
make test        # cargo test
make check       # fmt-check + clippy + test (the CI gate)
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

## Coding standards

- Run `cargo fmt` (enforced by CI).
- All clippy warnings are errors (`-D warnings`).
- Document public items with `///` comments.
- Prefer `Result` over panics for recoverable errors.
- `#![forbid(unsafe_code)]` where it is set — keep the crate unsafe-free.

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
