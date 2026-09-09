#!/usr/bin/env bash
# regen_dc_openmp.sh — regenerate an OpenMP variant of a `do concurrent` +
# OpenACC Fortran tree, in place.
#
# Three variants, selected by the first argument:
#
#   dc-openmp-target  (default)
#       !$acc data/reduction directives → !$omp target equivalents, but the
#       `do concurrent` loops are KEPT verbatim. The compiler maps them to the
#       device itself (NVHPC -stdpar=gpu, LLVM flang).
#
#   openmp
#       Everything dc-openmp-target does, PLUS dc_to_omp.py rewrites every
#       `do concurrent` as `!$omp target teams distribute parallel do`
#       (collapsing the header, mapping local()→private(), and stripping
#       `pure` from any procedure that ends up holding a target region — the
#       whole-program cascade). No `do concurrent` survives.
#
#   openmp-cpu
#       Like openmp, but both translators run with --target cpu: compute loops
#       become host `!$omp parallel do` (NOT `target teams distribute`, which is
#       flaky on some host fallbacks, notably GNU), and every device-data
#       directive becomes an inert no-op comment. Build it without offload.
#
# Runs in the repo root and modifies the working tree IN PLACE:
#   1. Lints `do concurrent` loops (dc_audit.py --strict).
#   2. Translates !$acc directives to !$omp via acc_to_omp.py --write.
#   3. (openmp variants only) Rewrites do-concurrent loops via dc_to_omp.py.
#   4. Applies every overlay patch under $DCPORT_PATCH_DIR/*/*.patch.
#   5. Runs the optional project hook tools/regen_post_hook.sh, if present.
#
# Use a clean checkout to call this script; do NOT call it from a branch that
# already has uncommitted source changes.
#
# Usage:
#   bash tools/regen_dc_openmp.sh                 # dc-openmp-target (default)
#   bash tools/regen_dc_openmp.sh dc-openmp-target
#   bash tools/regen_dc_openmp.sh openmp
#   bash tools/regen_dc_openmp.sh openmp-cpu
#
# An optional second argument sets the worker-process count for the per-file
# passes (0 = one per available core; default 1 = serial):
#   bash tools/regen_dc_openmp.sh openmp 0
#   DCPORT_JOBS=32 bash tools/regen_dc_openmp.sh openmp
#
# Project configuration (environment, all optional):
#   DCPORT_ROOTS      space-separated source roots to translate.
#                     Default: the ones of "src app benchmarks tests" that exist.
#   DCPORT_PATCH_DIR  overlay-patch directory. Default: patches/openmp
#   DCPORT_JOBS       worker processes (see above).
#
# Project-specific finishing steps — flipping a CMake default, swapping a
# toolchain file, editing a build preset — belong in tools/regen_post_hook.sh.
# It is sourced-and-run with $variant and $omp_target exported.
#
# Exits non-zero if any stage fails. Intended to be called from CI and
# (occasionally) by developers wanting to test a variant locally.

set -euo pipefail

variant="${1:-dc-openmp-target}"
# Worker processes for the per-file passes. Second positional arg, or
# DCPORT_JOBS. 0 = one per available core (respects SLURM binding);
# 1 (default) = serial.
jobs="${2:-${DCPORT_JOBS:-1}}"
omp_target=gpu          # acc_to_omp / dc_to_omp --target (gpu offload vs cpu host)
case "$variant" in
  dc-openmp-target) run_dc_to_omp=0 ;;
  openmp)           run_dc_to_omp=1 ;;
  openmp-cpu)       run_dc_to_omp=1; omp_target=cpu ;;
  *)
    echo "error: unknown variant '$variant'" >&2
    echo "       expected 'dc-openmp-target', 'openmp' or 'openmp-cpu'" >&2
    exit 2
    ;;
esac

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

# Source roots. Every directive-bearing directory must be listed: a root left
# out keeps its !$acc, which is inert on an OpenMP build, and — for the full
# `openmp` variant — is also excluded from the whole-program `pure` cascade.
if [[ -n "${DCPORT_ROOTS:-}" ]]; then
  read -r -a roots <<< "$DCPORT_ROOTS"
else
  roots=()
  for d in src app benchmarks tests; do
    [[ -d "$d" ]] && roots+=("$d")
  done
fi
if [[ ${#roots[@]} -eq 0 ]]; then
  echo "error: no source roots found; set DCPORT_ROOTS" >&2
  exit 2
fi

patch_dir="${DCPORT_PATCH_DIR:-patches/openmp}"

echo "==> variant: $variant  (translator --target $omp_target, -j $jobs)"
echo "==> roots:   ${roots[*]}"

echo "==> [1/5] dc_audit --strict"
python tools/dc_audit.py --strict -j "$jobs" "${roots[@]}"

echo "==> [2/5] acc_to_omp --write"
python tools/acc_to_omp.py --target "$omp_target" -j "$jobs" --write "${roots[@]}"

if [[ "$run_dc_to_omp" -eq 1 ]]; then
  echo "==> [3/5] dc_to_omp --write (full OpenMP: rewrite do concurrent loops)"
  # Runs AFTER acc_to_omp: the pure-strip cascade also sees the worksharing
  # regions acc_to_omp emitted from !$acc parallel loop (gpu: !$omp target;
  # cpu: !$omp parallel do), so pure is stripped consistently across both
  # directive sources. Same root set, so the cascade spans the whole program.
  python tools/dc_to_omp.py --target "$omp_target" -j "$jobs" --write "${roots[@]}"
else
  echo "==> [3/5] dc_to_omp SKIPPED (dc-openmp-target keeps do concurrent)"
fi

echo "==> [4/5] apply overlay patches ($patch_dir)"
shopt -s nullglob
patches=("$patch_dir"/*/*.patch "$patch_dir"/*.patch)
shopt -u nullglob
if [[ ${#patches[@]} -eq 0 ]]; then
  echo "    (no patches found under $patch_dir/)"
else
  for p in "${patches[@]}"; do
    echo "    applying $p"
    git apply --whitespace=nowarn "$p"
  done
fi

echo "==> [5/5] project post-hook"
if [[ -f tools/regen_post_hook.sh ]]; then
  echo "    running tools/regen_post_hook.sh"
  variant="$variant" omp_target="$omp_target" bash tools/regen_post_hook.sh
else
  echo "    (no tools/regen_post_hook.sh — skipped)"
fi

echo "==> regen complete ($variant)"
