# =============================================================================
# rnnoise-rs — Makefile
# =============================================================================

.DEFAULT_GOAL := help

VERSION := $(shell cat VERSION 2>/dev/null || echo 0.0.0)

.PHONY: help build test test-parity fmt fmt-check clippy spell doc deny bench reference clips check clean
help: ## Show this help message
	@awk 'BEGIN{FS=":.*##"; printf "\nrnnoise-rs  v$(VERSION)\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/{printf "  %-12s %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo ""

build: ## cargo build
	cargo build

test: ## cargo test
	cargo test

test-parity: ## cargo test against the C reference oracle (see scripts/c_reference.sh)
	./scripts/c_reference.sh
	RNNOISE_C_REFERENCE_DIR=.c_reference cargo test --all-features --test c_reference_parity -- --nocapture

fmt: ## cargo fmt
	cargo fmt --all

fmt-check: ## cargo fmt --check
	cargo fmt --all --check

clippy: ## cargo clippy with warnings denied
	cargo clippy --all-targets --all-features -- -D warnings

SPELL_GLOBS = "src/**/*.rs" "tests/**/*.rs" "benches/**/*.rs" "doc/**/*.md" "scripts/*.sh" \
	"examples/**/*.rs" "include/**/*.h" "Makefile" "README.md" "BENCHMARKS.md" "CHANGELOG.md" \
	"CONTRIBUTING.md" "AGENTS.md" "SECURITY.md"

spell: ## cspell gate (skips gracefully when cspell is not installed)
	@if command -v cspell >/dev/null 2>&1; then cspell --no-progress --config cspell.json $(SPELL_GLOBS); else echo "spell: cspell not installed — skipping (run: npx cspell)"; fi

doc: ## cargo doc
	cargo doc --no-deps

deny: ## cargo deny (advisories, licenses, bans, sources) — config in deny.toml
	cargo deny check

bench: ## criterion benchmarks, results saved as a baseline for CI artifacts
	cargo bench --all-features --bench process -- --save-baseline ci

reference: ## build the pinned Xiph C reference (scalar _c path) and run the clips
	./scripts/c_reference.sh

clips: ## regenerate the deterministic test clips in test_data/clips/
	python3 scripts/make_clips.py

check: fmt-check clippy spell test deny ## fmt + clippy + spell + test + deny (the CI gate)

clean: ## cargo clean
	cargo clean
