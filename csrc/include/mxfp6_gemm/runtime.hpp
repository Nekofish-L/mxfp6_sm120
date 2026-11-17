#pragma once
#include <ATen/ATen.h>
#include "mxfp6_gemm/types.hpp"
#include "mxfp_common/workspace.hpp"

namespace mxfp6_native {
using mxfp6_dispatch::RasterOrder;
using mxfp6_dispatch::round_up;
using mxfp6_dispatch::is_target_nk;
using mxfp_common::WorkspaceLayout;
using mxfp_common::WorkspaceSelection;
using mxfp_common::get_workspace_layout;
struct MXFP6WorkspaceTag {};
extern mxfp_common::WorkspacePool<MXFP6WorkspaceTag> workspace_pool;
bool stream_k_enabled();
void check_swizzle(int64_t value);
RasterOrder parse_raster_order(int64_t value);
// Kernel selection is synchronous on the calling host thread. A scoped
// thread-local keeps output dtype out of the large policy call graph while
// remaining safe for concurrent launches from independent Python threads.
extern thread_local at::ScalarType current_output_dtype;

class OutputDtypeGuard {
 public:
  explicit OutputDtypeGuard(at::ScalarType output_dtype)
      : previous_(current_output_dtype) {
    current_output_dtype = output_dtype;
  }

  ~OutputDtypeGuard() {
    current_output_dtype = previous_;
  }

 private:
  at::ScalarType previous_;
};

class WorkspaceCollectionGuard {
 public:
  explicit WorkspaceCollectionGuard(bool enabled)
      : previous_(workspace_pool.collection_enabled()) {
    workspace_pool.collection_enabled() = enabled;
  }

  ~WorkspaceCollectionGuard() {
    workspace_pool.collection_enabled() = previous_;
  }

 private:
  bool previous_;
};

at::Tensor launch_w6a6_policy(at::Tensor const& a, at::Tensor const& b, at::Tensor const& sfa, at::Tensor const& sfb, int64_t m, int64_t n, int64_t k, double alpha, int device_index, int sm_count);
at::Tensor launch_w6a8_policy(at::Tensor const& a, at::Tensor const& b, at::Tensor const& sfa, at::Tensor const& sfb, int64_t m, int64_t n, int64_t k, double alpha, int device_index, int sm_count, bool use_pdl);
at::Tensor launch_w6a8_config(at::Tensor const& a, at::Tensor const& b, at::Tensor const& sfa, at::Tensor const& sfb, int64_t m, int64_t n, int64_t k, double alpha, int device_index, int sm_count, int64_t config_id, int64_t swizzle, int64_t raster_order, bool use_pdl);
} // namespace mxfp6_native
