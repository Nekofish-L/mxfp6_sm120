#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
bash scripts/apply_cutlass_patches.sh --runtime-only
exec "${PYTHON:-python3}" -m pip wheel . \
  --no-build-isolation --no-deps --wheel-dir dist "$@"
