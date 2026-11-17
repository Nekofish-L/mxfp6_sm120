#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>

namespace mxfp6_native {
// Filled by the existing argument builder during prepare; no launch occurs.
struct PreparedLaunch {
  std::function<void(cudaStream_t, bool)> run;
  std::function<void(cudaStream_t)> reset_workspace;
  // Alias the owned, correctly aligned CUTLASS Params; no serialized pointer
  // escapes its owner and no TMA descriptor is rebuilt on a hot call.
  std::shared_ptr<void> parameters;
  size_t parameter_bytes = 0;
  dim3 grid, block;
  int shared_bytes = 0;
  bool used = false;
  at::Tensor workspace;
};
extern thread_local PreparedLaunch* preparing;
} // namespace mxfp6_native
