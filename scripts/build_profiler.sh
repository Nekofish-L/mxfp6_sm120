#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-w6a6}"

case "${MODE}" in
  w6a6)
    BUILD_DIR="${ROOT}/build_profiler"
    KERNELS="cutlass3x_sm120_bstensorop_gemm_ue8m0xe3m2_ue8m0xe3m2_f32_void_f16_*"
    ;;
  w6a8)
    BUILD_DIR="${ROOT}/build_profiler_a8b6"
    KERNELS="cutlass3x_sm120_bstensorop_gemm_ue8m0xe4m3_ue8m0xe3m2_f32_void_f16_*"
    ;;
  *)
    echo "usage: $0 [w6a6|w6a8]" >&2
    exit 2
    ;;
esac

"${ROOT}/scripts/apply_cutlass_patches.sh"
cmake -S "${ROOT}/third_party/cutlass" -B "${BUILD_DIR}" -G Ninja \
  -DCUTLASS_NVCC_ARCHS=120a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm \
  -DCUTLASS_LIBRARY_KERNELS="${KERNELS}" \
  -DCUTLASS_LIBRARY_MINIMAL_MXFP6_PROFILER=ON \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON \
  -DCUTLASS_LIBRARY_INSTANTIATION_LEVEL=max \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --parallel "${MAX_JOBS:-$(nproc)}"
