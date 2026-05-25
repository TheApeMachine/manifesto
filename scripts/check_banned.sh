#!/usr/bin/env bash
# scripts/check_banned.sh — mechanical enforcement of AGENTS.md banned patterns
# and ARCHITECTURE.md §7 invariants.
#
# Exits 0 if clean, 1 if any violation found. Lists every violation with
# file:line so the agent making the change has a concrete fix list.
#
# Run via `make check`. This script is the canary that catches the
# recurring shortcuts before they land. Do not loosen its rules without
# updating AGENTS.md and ARCHITECTURE.md first.

set -u

# Move to repo root.
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$0"/..)" || exit 2

violations=0

fail() {
    printf '  %s\n' "$1" >&2
    violations=$((violations + 1))
}

section() {
    printf '\n=== %s ===\n' "$1"
}

# -----------------------------------------------------------------------------
# 1. Zero-host-sync (ARCHITECTURE.md §2.2)
# device.Backend methods must write to `dst unsafe.Pointer`, never return Go
# scalars. Reductions, dot products, losses, sampling, similarity — all of
# them write to device-resident memory.
# -----------------------------------------------------------------------------
section "1. Zero-host-sync (§2.2): no scalar returns on device.Backend"

if [ -f device/interface.go ]; then
    # Single-line method declarations: `Foo(args) float32`.
    while IFS= read -r line; do
        fail "device/interface.go: scalar return violates §2.2 — $line"
    done < <(grep -nE '^\s*[A-Z][A-Za-z0-9_]*\s*\([^)]*\)\s*(float32|float64|int32|int64|uint32|uint64|bool)\s*$' device/interface.go || true)

    # Multi-line method declarations: any closing `) float32` (etc.) line
    # whose preceding context inside the interface block is a method
    # opening. We catch the closing line and report its file:line.
    while IFS= read -r line; do
        fail "device/interface.go: scalar return violates §2.2 (multi-line) — $line"
    done < <(grep -nE '^\s*\)\s*(float32|float64|int32|int64|uint32|uint64|bool)\s*$' device/interface.go || true)
fi

# -----------------------------------------------------------------------------
# 2. Dtype-prefixed filenames (ARCHITECTURE.md §2.3 line 202)
# Files under device/cpu/<family>/ must not be named f32_*, f16_*, fp16_*,
# f64_*, bf16_*. Domain split is semantic (e.g. "standard", "gated", "math"),
# never by dtype. Exception: int8_/int4_ in dequant/ and quant/ because those
# name a quantization SCHEME, not a floating dtype (spec line 1126).
# -----------------------------------------------------------------------------
section "2. Dtype-prefixed filenames (§2.3)"

if [ -d device/cpu ]; then
    while IFS= read -r path; do
        fail "dtype-prefixed filename: $path"
    done < <(find device/cpu -type f \( -name "*.go" -o -name "*.s" \) 2>/dev/null \
        | grep -E '/(f32|f16|fp16|f64|bf16)_[A-Za-z0-9_]+\.(go|s)$' \
        | grep -vE '/(dequant|quant)/' \
        || true)
fi

# -----------------------------------------------------------------------------
# 3. Catch-all backend files (ARCHITECTURE.md §7)
# No device_missing*, device_remaining*, *_stub_ops*, *_extra* files that
# aggregate unrelated interface methods. Anti-patterns from §2.3.
# -----------------------------------------------------------------------------
section "3. Catch-all backend files (§7)"

while IFS= read -r path; do
    fail "catch-all file: $path"
done < <(find device -type f \( \
    -name 'device_missing*' -o \
    -name 'device_remaining*' -o \
    -name '*_stub_ops*' -o \
    -name '*_extra*' \
\) 2>/dev/null || true)

# -----------------------------------------------------------------------------
# 4. Root forwarding shims (ARCHITECTURE.md §2.3 anti-patterns)
# device/<backend>/backend_<family>.go forwarding shims are forbidden.
# Embedded family types promote their methods directly.
# Allow-list: backend.go, backend_config.go.
# -----------------------------------------------------------------------------
section "4. Root forwarding shims (§2.3)"

for backend in cpu metal cuda xla; do
    [ -d "device/$backend" ] || continue
    while IFS= read -r path; do
        case "$(basename "$path")" in
            # Allow-list: legitimate root files.
            #   backend.go              — the Backend struct definition.
            #   backend_config.go       — the Config struct.
            #   backend_stub*.go        — build-tag stubs.
            #   backend_test.go         — root-level tests.
            #   backend_new_*.go        — build-tag-split NewBackend constructor.
            backend.go|backend_config.go|backend_stub*.go|backend_test.go|backend_new_*.go) ;;
            *) fail "root forwarding shim (or root file outside §2.3 layout): $path" ;;
        esac
    done < <(find "device/$backend" -maxdepth 1 -name 'backend_*.go' 2>/dev/null || true)
done

# -----------------------------------------------------------------------------
# 5. Model-specific Go shortcuts (manifest-first principle)
# Every model architecture compiles from YAML manifests over atomic ops. No
# Go packages or files dedicated to a specific model family. The diffusion-Go
# fast-path that motivated this rule is documented in GAPS.md §6.5.
# -----------------------------------------------------------------------------
section "5. Model-specific Go shortcuts (manifest-first)"

model_pkgs='manifesto/(diffusion|llama|bert|sd3|sdxl|flux|dit|stable_diffusion|stablediffusion|unet|vae)'
while IFS= read -r line; do
    fail "model-specific package import: $line"
done < <(grep -rnE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    "\"github\\.com/theapemachine/$model_pkgs\"" . 2>/dev/null || true)

# Also catch architecture-named Go files: llama.go, flux.go, sd3.go in
# packages that should be generic. Only flag if they are not in clearly
# scoped locations (e.g. asset templates, test fixtures).
while IFS= read -r path; do
    case "$path" in
        */asset/*|*/testdata/*|*/.git/*) ;;
        *) fail "model-named Go file (should be a YAML recipe in manifesto/asset/): $path" ;;
    esac
done < <(find . -type f -name '*.go' 2>/dev/null \
    | grep -iE '/(diffusion|denoise|unet|vae_specific|flux_specific|sd3_specific|llama_specific)\.go$' \
    || true)

# -----------------------------------------------------------------------------
# 6. Go-heap workspace allocations (ARCHITECTURE.md §5.2)
# The execution workspace must live outside the Go GC heap. No
# `make([]byte, workspaceSize)` for the workspace. Allocate via
# posix_memalign / cudaMalloc / MTLBuffer / PjRtBuffer.
# -----------------------------------------------------------------------------
section "6. Go-heap workspace allocation (§5.2)"

while IFS= read -r line; do
    fail "Go-heap workspace alloc — workspace must be off-GC-heap: $line"
done < <(grep -rnE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    'make\(\[\]byte.*[wW]orkspace' . 2>/dev/null || true)

# -----------------------------------------------------------------------------
# 7. runtime.Pinner in async dispatch (ARCHITECTURE.md §5.2)
# Native driver completion callbacks (cudaLaunchHostFunc,
# addCompletedHandler) run on non-Go threads. runtime.Pinner from those
# paths risks deadlock during stop-the-world GC. If workspace is off-heap,
# pinning is unnecessary anyway.
# -----------------------------------------------------------------------------
section "7. runtime.Pinner (§5.2)"

while IFS= read -r line; do
    fail "runtime.Pinner — verify not in async dispatch path: $line"
done < <(grep -rnE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    'runtime\.Pinner' . 2>/dev/null || true)

# -----------------------------------------------------------------------------
# 8. Banned phrases (AGENTS.md §1)
# The phrases that mark a shortcut about to ship.
# -----------------------------------------------------------------------------
section "8. Banned phrases (AGENTS.md §1)"

# Match the exact banned phrases inside Go comments. Case-insensitive.
# We deliberately do NOT match "approximation" alone since legitimate code
# may describe approximation theory; we match the exact disclaimer phrasing.
phrases='for now|approximation acceptable|required vs optional backend|fallback to Go|TODO.*later|will implement.*later|placeholder.*until'
while IFS= read -r line; do
    fail "banned phrase: $line"
done < <(grep -rniE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    "(//|/\\*).*($phrases)" . 2>/dev/null || true)

# -----------------------------------------------------------------------------
# 9. Backend conformance assertion (ARCHITECTURE.md §2.1, AGENTS.md §1)
# Every backend must have `var _ device.Backend = (*Backend)(nil)` so the
# compiler catches missing methods. This is the canary for closed-world.
# -----------------------------------------------------------------------------
section "9. Backend conformance assertion"

for backend in cpu metal cuda xla; do
    if [ ! -d "device/$backend" ]; then
        fail "device/$backend: directory missing"
        continue
    fi
    # The assertion may live in any .go file in the backend root.
    if ! grep -rqE '_\s*device\.Backend\s*=\s*\(\*Backend\)\(nil\)' "device/$backend"/*.go 2>/dev/null; then
        fail "device/$backend: missing \`var _ device.Backend = (*Backend)(nil)\` static assertion in root .go files"
    fi
done

# -----------------------------------------------------------------------------
# 10. Top-level orphan kernels (ARCHITECTURE.md §2.3 anti-patterns)
# Metal/CUDA kernels must live inside a family subdirectory. No bare
# *.metal or *.cu at the backend root.
# -----------------------------------------------------------------------------
section "10. Orphan kernel files at backend root"

if [ -d device/metal ]; then
    while IFS= read -r path; do
        fail "orphan Metal kernel at backend root: $path"
    done < <(find device/metal -maxdepth 1 -type f -name '*.metal' 2>/dev/null || true)
fi

if [ -d device/cuda ]; then
    while IFS= read -r path; do
        fail "orphan CUDA kernel at backend root: $path"
    done < <(find device/cuda -maxdepth 1 -type f -name '*.cu' 2>/dev/null || true)
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
printf '\n'
if [ "$violations" -gt 0 ]; then
    printf 'FAILED: %d banned-pattern violation(s)\n' "$violations" >&2
    printf 'See AGENTS.md and ARCHITECTURE.md for the rules. See GAPS.md for known gaps.\n' >&2
    exit 1
fi
printf 'OK: no banned-pattern violations\n'
