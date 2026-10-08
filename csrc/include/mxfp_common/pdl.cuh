#pragma once
#include <cuda_runtime.h>
#include "cutlass/arch/grid_dependency_control.h"

namespace mxfp_common {
// Call after independent setup and before reading a predecessor's output.
// Releasing a successor does not make this grid's stores visible; its wait
// still has to complete before it consumes them.
__device__ __forceinline__ void dependent_prologue(bool enabled) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  static_assert(cutlass::arch::IsGdcGloballyEnabled,
                "PDL kernels require CUTLASS_ENABLE_GDC_FOR_SM100");
#endif
  if (enabled) {
    cutlass::arch::wait_on_dependent_grids();
    if (threadIdx.x == 0) cutlass::arch::launch_dependent_grids();
  }
}

template<class Kernel, class... Args>
cudaError_t launch_dependent_kernel(bool enabled, Kernel kernel, dim3 grid,
                                  int threads, size_t shared,
                                  cudaStream_t stream, Args... args) {
  if (!enabled) {
    kernel<<<grid, threads, shared, stream>>>(args...);
    return cudaGetLastError();
  }
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(threads);
  config.dynamicSmemBytes = shared;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, kernel, args...);
}
} // namespace mxfp_common
