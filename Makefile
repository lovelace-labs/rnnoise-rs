# =============================================================================
# rnnoise-rs — Makefile
# =============================================================================

.DEFAULT_GOAL := help

VERSION := $(shell cat VERSION 2>/dev/null || echo 0.0.0)

.PHONY: help build test fmt fmt-check clippy doc check clean
help: ## Show this help message
	@awk 'BEGIN{FS=":.*##"; printf "\nrnnoise-rs  v$(VERSION)\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/{printf "  %-12s %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo ""

build: ## cargo build
	cargo build

test: ## cargo test
	cargo test

fmt: ## cargo fmt
	cargo fmt --all

fmt-check: ## cargo fmt --check
	cargo fmt --all --check

clippy: ## cargo clippy with warnings denied
	cargo clippy --all-targets -- -D warnings

doc: ## cargo doc
	cargo doc --no-deps

check: fmt-check clippy test ## fmt + clippy + test (the CI gate)

clean: ## cargo clean
	cargo clean
