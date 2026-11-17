#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include "mxfp8_gemm/pdl.cuh"
#include "mxfp8_gemm/validation.hpp"

namespace mxfp8_host {
// Generated descriptors own launch constants; device templates remain unchanged.
template<int BM, int BN, int BK, int Stages, bool Swap, int WarpM,
         bool PackedSF, int Threads, int SharedBytes>
struct Config {
  static constexpr int bm = BM, bn = BN, bk = BK, stages = Stages;
  static constexpr bool swap = Swap, packed_sf = PackedSF;
  static constexpr int warp_m = WarpM, threads = Threads, shared_bytes = SharedBytes;
};

template<class Kernel, class C, int FixedK>
void launch(at::Tensor const& a, at::Tensor const& b, at::Tensor const& sa,
            at::Tensor const& sb, at::Tensor const& out) {
  int m = C::swap ? b.size(0) : a.size(0);
  int n = C::swap ? a.size(0) : b.size(0), k = a.size(1);
  auto fn = Kernel::template get<C, FixedK>();
  if constexpr (C::shared_bytes >= 48 * 1024) {
    auto err = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    C::shared_bytes);
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  }
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  bool pdl = mxfp8_runtime::pdl_enabled();
  auto err = mxfp8_runtime::launch_dependent_kernel(
      pdl, fn, dim3((n + C::bn - 1) / C::bn, (m + C::bm - 1) / C::bm),
      C::threads, C::shared_bytes, stream,
      static_cast<const uint8_t*>((C::swap ? b : a).data_ptr()),
      static_cast<const uint8_t*>((C::swap ? a : b).data_ptr()),
      static_cast<const uint8_t*>((C::swap ? sb : sa).data_ptr()),
      static_cast<const uint8_t*>((C::swap ? sa : sb).data_ptr()),
      static_cast<__nv_bfloat16*>(out.data_ptr()), m, n, k, pdl);
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

template<class Kernel, class C>
void dispatch(at::Tensor const& a, at::Tensor const& b, at::Tensor const& sa,
              at::Tensor const& sb, at::Tensor const& out) {
  if constexpr (Kernel::specialize_k) {
    if (a.size(1) == 2048) return launch<Kernel, C, 2048>(a, b, sa, sb, out);
    if (a.size(1) == 2560) return launch<Kernel, C, 2560>(a, b, sa, sb, out);
  }
  launch<Kernel, C, 0>(a, b, sa, sb, out);
}
} // namespace mxfp8_host
