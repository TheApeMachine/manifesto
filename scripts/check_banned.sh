#!/usr/bin/env bash
# scripts/check_banned.sh — mechanical enforcement of manifest-first contract.
#
# manifesto is the compiler/optimizer/codegen/scheduler pipeline (the spec
# in /puter/ARCHITECTURE.md). Its banned patterns target the recurring
# shortcut: AI agents writing per-model Go fast-paths instead of composing
# atomic ops via YAML manifests. The diffusion-Go contamination documented
# in /puter/GAPS.md §6.5 is exactly what these checks prevent.
#
# Exits 0 if clean, 1 if any violation found.

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$0"/..)" || exit 2

violations=0
fail() { printf '  %s\n' "$1" >&2; violations=$((violations + 1)); }
section() { printf '\n=== %s ===\n' "$1"; }

# -----------------------------------------------------------------------------
# 1. No model-specific Go packages.
# manifesto is the closed-world compiler. Model architectures are YAML
# recipes under asset/template/model/, not Go packages. The presence of
# manifesto/diffusion/ is the named anti-pattern (GAPS.md §6.5).
# -----------------------------------------------------------------------------
section "1. Model-specific Go packages (manifest-first)"

# Known allow-listed top-level dirs. Anything else that matches a model
# name pattern is a violation.
banned_dirs='diffusion llama bert sd3 sdxl flux dit stable_diffusion stablediffusion unet vae gpt mistral qwen gemma'
for dir in $banned_dirs; do
    if [ -d "$dir" ]; then
        fail "model-specific Go package: ./$dir (should be YAML under asset/template/model/)"
    fi
done

# -----------------------------------------------------------------------------
# 2. No imports of model-specific packages from within manifesto.
# -----------------------------------------------------------------------------
section "2. Imports of model-specific subpackages"

model_pkgs='manifesto/(diffusion|llama|bert|sd3|sdxl|flux|dit|stable_diffusion|stablediffusion|unet|vae)'
while IFS= read -r line; do
    fail "model-specific import: $line"
done < <(grep -rnE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    "\"github\\.com/theapemachine/$model_pkgs\"" . 2>/dev/null || true)

# -----------------------------------------------------------------------------
# 3. No model-named files in runtime/ or compiler/.
# runtime/executor_diffusion.go, runtime/scheduler.go (the FlowMatch one),
# runtime/latents.go are the contamination. The general rule: no file in
# runtime/ or compiler/ may be named after a model architecture or a
# diffusion-specific concept.
# -----------------------------------------------------------------------------
section "3. Model-named files in runtime/ or compiler/"

model_file_pattern='(diffusion|latent|denoise|sigma|timestep|flow_match|euler_discrete|unet|vae|sd3|sdxl|flux|llama|bert)'

for dir in runtime compiler optimizer codegen scheduler; do
    [ -d "$dir" ] || continue
    while IFS= read -r path; do
        # Spec's §4.4 DAG scheduler is a legitimate "scheduler" — the file
        # only earns a violation when it's named after a diffusion concept
        # (flow_match, euler, sigma, timestep, latent, denoise).
        case "$path" in
            */scheduler.go|*/dag_scheduler.go|*/stream_scheduler.go) ;;
            *) fail "model-named file (should be YAML or a generic primitive): $path" ;;
        esac
    done < <(find "$dir" -maxdepth 1 -type f -name '*.go' 2>/dev/null \
        | grep -iE "$model_file_pattern" || true)
done

# -----------------------------------------------------------------------------
# 4. Step-name dispatch for model-specific operations.
# The executor must not have `case "diffusion.prepare_latents":` or
# `case "scheduler.timesteps":` — those are model-specific. Atomic step
# names are domain-prefixed by op family (math.*, attention.*, etc.).
# -----------------------------------------------------------------------------
section "4. Model-specific step-name dispatch"

while IFS= read -r line; do
    fail "model-specific step case: $line"
done < <(grep -rnE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    'case\s+"(diffusion\.|scheduler\.timesteps|scheduler\.bind_latents|scheduler\.delta)' \
    . 2>/dev/null || true)

# -----------------------------------------------------------------------------
# 5. YAML kind: format
# Every YAML in asset/ uses the new `kind:` format on the first non-empty,
# non-comment line. Old format must be migrated. (See GAPS.md §6.5 list.)
# -----------------------------------------------------------------------------
section "5. YAML kind: format in asset/"

if [ -d asset ]; then
    while IFS= read -r path; do
        first_real=$(awk 'NF && !/^[[:space:]]*#/ { print; exit }' "$path")
        case "$first_real" in
            kind:*) ;;
            "") ;;
            *) fail "old-format YAML (must start with \`kind:\`): $path" ;;
        esac
    done < <(find asset -type f \( -name '*.yml' -o -name '*.yaml' \) 2>/dev/null)
fi

# -----------------------------------------------------------------------------
# 6. Banned phrases (mirror puter/AGENTS.md §1)
# -----------------------------------------------------------------------------
section "6. Banned phrases"

phrases='for now|approximation acceptable|required vs optional backend|fallback to Go|TODO.*later|will implement.*later|placeholder.*until'
while IFS= read -r line; do
    fail "banned phrase: $line"
done < <(grep -rniE --include='*.go' --exclude-dir=vendor --exclude-dir=.git \
    "(//|/\\*).*($phrases)" . 2>/dev/null || true)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
printf '\n'
if [ "$violations" -gt 0 ]; then
    printf 'FAILED: %d banned-pattern violation(s)\n' "$violations" >&2
    printf 'See puter/AGENTS.md and puter/GAPS.md for the rules and known gaps.\n' >&2
    exit 1
fi
printf 'OK: no banned-pattern violations\n'
