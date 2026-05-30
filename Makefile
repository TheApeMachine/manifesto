.PHONY: test check verify test-jit test-jit-verify test-metal test-metal-verify

# CGO is required for Metal MTLLibrary compilation on Darwin.
METAL_TAGS := -tags=cgo

# The pool package uses go:linkname to access runtime scheduling
# primitives (dropg, readgstatus) for zero-overhead goroutine parking.
# Go 1.26 restricts these by default; -checklinkname=0 preserves access.
LDFLAGS := -ldflags='-checklinkname=0'

# LLVM JIT tests require a system LLVM (Homebrew: brew install llvm). Override
# LLVM_CONFIG when llvm-config is not on PATH (common on macOS):
#   make test-jit LLVM_CONFIG=/opt/homebrew/opt/llvm/bin/llvm-config
ifeq ($(shell command -v llvm-config 2>/dev/null),)
  ifneq ($(wildcard /opt/homebrew/opt/llvm/bin/llvm-config),)
    LLVM_CONFIG := /opt/homebrew/opt/llvm/bin/llvm-config
  endif
endif
LLVM_CONFIG ?= llvm-config
LLVM_VERSION_MAJOR := $(shell $(LLVM_CONFIG) --version 2>/dev/null | cut -d. -f1)
LLVM_TAGS := -tags=codegen_llvm,llvm$(LLVM_VERSION_MAJOR)
LLVM_CFLAGS := $(shell $(LLVM_CONFIG) --cflags 2>/dev/null)
LLVM_LDFLAGS := $(shell $(LLVM_CONFIG) --ldflags --libs core mcjit native 2>/dev/null) -lm
export CGO_CFLAGS := $(LLVM_CFLAGS)
export CGO_CPPFLAGS := $(LLVM_CFLAGS)
export CGO_CXXFLAGS := $(LLVM_CFLAGS) -std=c++17
export CGO_LDFLAGS := $(LLVM_LDFLAGS)

DUMP ?= manifesto.txt

# check runs mechanical enforcement of manifest-first contract.
# See puter/AGENTS.md and puter/GAPS.md §6.5 for the rules.
check:
	@bash "$(CURDIR)/scripts/check_banned.sh"

test:
	go test $(LDFLAGS) -v ./...

# test-jit runs LLVM MCJIT parity tests (Phase 3.2a). Requires LLVM dev libs.
test-jit:
	@test -n "$(LLVM_VERSION_MAJOR)" || (echo "LLVM not found; install llvm and set LLVM_CONFIG" && exit 1)
	go test $(LDFLAGS) $(LLVM_TAGS) -v ./codegen/...

test-jit-verify: check test-jit

# test-metal runs MTLLibrary fusion parity tests (Phase 3.2b). Requires Darwin + Metal.
test-metal:
	CGO_ENABLED=1 go test $(LDFLAGS) $(METAL_TAGS) -v ./codegen/...

test-metal-verify: check test-metal

# verify is the gate: banned-pattern check first, then tests.
verify: check test

dump:
	python3 "$(CURDIR)/scripts/dump-repo.py" "$(DUMP)"