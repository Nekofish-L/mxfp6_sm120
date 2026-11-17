#pragma once
#include <torch/custom_class.h>
#include "mxfp6_gemm/prepared.hpp"
#include "mxfp6_gemm/types.hpp"

namespace mxfp6_native {
class PreparedW6A8 : public torch::CustomClassHolder {
  PreparedLaunch launch_;
  std::vector<at::Tensor> tensors_;
  struct StorageSnapshot {
    void* pointer;
    std::vector<int64_t> sizes, strides;
    at::ScalarType dtype;
    c10::Device device;
  };
  std::vector<StorageSnapshot> snapshots_;
  mxfp6_dispatch::Config config_;
  bool pdl_;
 public:
  PreparedW6A8(at::Tensor a, at::Tensor b, at::Tensor sfa, at::Tensor sfb,
              int64_t m, int64_t n, int64_t k, double alpha,
              int64_t config_id, int64_t swizzle, int64_t raster_order,
              at::ScalarType dtype, bool pdl);
  at::Tensor run();
  std::vector<int64_t> config() const {
    return {config_.config_id, config_.swizzle, config_.raster_order};
  }
  PreparedLaunch& launch() { return launch_; }
  int device_index() const { return snapshots_[0].device.index(); }
  at::ScalarType output_dtype() const { return snapshots_.back().dtype; }
};
} // namespace mxfp6_native
