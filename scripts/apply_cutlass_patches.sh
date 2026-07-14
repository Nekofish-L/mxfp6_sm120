#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-${ROOT}/third_party/cutlass}"
EXPECTED_COMMIT="e6233cbac5d7c7a865c19c91cd684ceece19513c"

mode=apply
runtime_only=0
for arg in "$@"; do
  case "${arg}" in
    --check) mode=check ;;
    --reverse) mode=reverse ;;
    --runtime-only) runtime_only=1 ;;
    *)
      echo "usage: $0 [--check|--reverse] [--runtime-only]" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CUTLASS_DIR}/include/cutlass/cutlass.h" ]]; then
  echo "CUTLASS is not initialized at ${CUTLASS_DIR}." >&2
  echo "Run: git submodule update --init --depth 1 third_party/cutlass" >&2
  exit 1
fi

# Source archives do not contain submodule Git metadata. In a Git checkout,
# enforce the exact upstream revision before touching the patch queue.
if [[ -e "${CUTLASS_DIR}/.git" ]]; then
  actual_commit="$(git -C "${CUTLASS_DIR}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "CUTLASS commit mismatch: expected ${EXPECTED_COMMIT}, got ${actual_commit}." >&2
    exit 1
  fi
fi

patches=(
  "${ROOT}/patches/cutlass/0001-sm120-mxfp6-small-tile-runtime.patch"
)
if [[ "${runtime_only}" -eq 0 ]]; then
  patches+=("${ROOT}/patches/cutlass/0002-sm120-mxfp6-profiler-search.patch")
fi

patch_is_applied() {
  git -C "${CUTLASS_DIR}" apply --check --reverse "$1" >/dev/null 2>&1
}

patch_is_applicable() {
  git -C "${CUTLASS_DIR}" apply --check "$1" >/dev/null 2>&1
}

if [[ "${mode}" == check ]]; then
  for patch in "${patches[@]}"; do
    if patch_is_applied "${patch}"; then
      echo "applied: $(basename "${patch}")"
    elif patch_is_applicable "${patch}"; then
      echo "not applied: $(basename "${patch}")"
      exit 1
    else
      echo "conflict: $(basename "${patch}")" >&2
      exit 1
    fi
  done
  exit 0
fi

if [[ "${mode}" == reverse ]]; then
  for ((i=${#patches[@]} - 1; i>=0; --i)); do
    patch="${patches[i]}"
    if patch_is_applied "${patch}"; then
      git -C "${CUTLASS_DIR}" apply --reverse "${patch}"
      echo "reversed: $(basename "${patch}")"
    elif patch_is_applicable "${patch}"; then
      echo "already clean: $(basename "${patch}")"
    else
      echo "cannot reverse cleanly: $(basename "${patch}")" >&2
      exit 1
    fi
  done
  exit 0
fi

for patch in "${patches[@]}"; do
  if patch_is_applied "${patch}"; then
    echo "already applied: $(basename "${patch}")"
  elif patch_is_applicable "${patch}"; then
    git -C "${CUTLASS_DIR}" apply "${patch}"
    echo "applied: $(basename "${patch}")"
  else
    echo "cannot apply cleanly: $(basename "${patch}")" >&2
    exit 1
  fi
done
