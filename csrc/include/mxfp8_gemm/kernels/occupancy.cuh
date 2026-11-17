#pragma once
#include "mxfp8_gemm/kernels/cp_async_body.cuh"

namespace mxfp8_occupancy {
template<int BM,int BN,int BK,int Stages,bool Swap,int FixedK=0,int WarpM=2,bool PackedSF=false>
__global__ __launch_bounds__(128) void gemm(
    const uint8_t* __restrict__ a,const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ sa,const uint8_t* __restrict__ sb,
    __nv_bfloat16* __restrict__ out,int m,int n,int dynamic_k,bool pdl) {
  using namespace mxfp8_cp_async;
  constexpr int Warps=4;
  constexpr bool CacheAll=false;
  constexpr Store StorePolicy=Store::VectorSeparate;
#include "mxfp8_gemm/kernels/cp_async_body.inc"
}
} // namespace mxfp8_occupancy
