#pragma once
#include "mxfp8_gemm/layout.cuh"
#include "mxfp8_gemm/pdl.cuh"
// Diagnostic alternative to the 384-thread CUTLASS warp-specialized templates.
// Uses SM120 warp-level CuTe MMA with 2/3/5/6/7 warps and TMA,
// independent CTAs, and direct BF16 stores. No input repacking is required.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include "cute/arch/mma_sm120.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/arch/barrier.h"
#include "mxfp8_gemm/validation.hpp"

namespace mxfp8_tma_oddwarp {
using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,cutlass::float_ue8m0_t,32>;

using mxfp8_layout::sf_index;
using mxfp8_layout::tma::sm_index;
__device__ __forceinline__ void copy16(uint8_t* dst,const uint8_t* src,bool valid) {
  uint32_t addr=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
               "r"(addr),"l"(src),"r"(valid?16:0));
}
__device__ __forceinline__ void copy4(uint8_t* dst,const uint8_t* src,bool valid) {
  uint32_t addr=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
               "r"(addr),"l"(src),"r"(valid?4:0));
}
template<int BM,int BN,int BK,int Warps>
__device__ void prefetch(uint8_t* sm,const uint8_t* a,const uint8_t* b,
                         const uint8_t* sa,const uint8_t* sb,
                         int pm,int pn,int start,int m,int n,int k, CUtensorMap const* ma, CUtensorMap const* mb, uint64_t* barrier) {
  static_assert(BK%128==0);
  if(threadIdx.x==0) {
    cute::set_barrier_transaction_bytes(*barrier,(BM+BN)*BK);
    cute::SM90_TMA_LOAD_3D::copy(ma,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm,0,pm,start/128);
    cute::SM90_TMA_LOAD_3D::copy(mb,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm+BM*BK,0,pn,start/128);
  }
#pragma unroll
  for(int i=threadIdx.x;i<(BM+BN)*(BK/128);i+=Warps*32) {
    int row=i/(BK/128),g=(i%(BK/128))*4;
    bool is_a=row<BM;
    int r=is_a?pm+row:pn+row-BM;
    bool valid=r<(is_a?m:n) && start+g*32<k;
    const uint8_t* src=(is_a?sa:sb)+(valid?sf_index(r,start/32+g,k):0);
    copy4(sm+(BM+BN)*BK+row*(BK/32)+g,src,valid);
  }
  asm volatile("cp.async.commit_group;");
}
template<int BM,int BN,int BK,int Stages,bool Swap,int FixedK=0,int WarpM=2,bool PackedSF=false,int Warps=4>
__global__ __launch_bounds__(Warps*32) void gemm(
    const uint8_t* __restrict__ a,const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ sa,const uint8_t* __restrict__ sb,
    __nv_bfloat16* __restrict__ out,int m,int n,int dynamic_k, const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, bool pdl) {
  constexpr bool ReserveOutput=false;
#include "mxfp8_gemm/kernels/tma_warp_body.inc"
}

#include "mxfp8_gemm/tma_warp_launch.inc"
void mm_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,
            at::Tensor const& sb,at::Tensor const& out,int64_t config) {
  mxfp8_common::validate(a,b,sa,sb,out);
  c10::cuda::CUDAGuard guard(a.device());
#define RUN(ID,...) case ID:specialized<__VA_ARGS__>(a,b,sa,sb,out);break
  switch(config) {
    RUN(0,32,32,256,2,true,2,true,2);
    RUN(1,32,64,256,1,true,2,true,2);
    RUN(2,48,16,256,2,true,3,true,3);
    RUN(3,48,32,256,1,true,3,true,3);
    RUN(4,48,64,128,2,true,3,true,3);
    RUN(5,80,16,256,1,true,5,true,5);
    RUN(6,80,32,256,1,true,5,true,5);
    RUN(7,80,32,128,2,true,5,true,5);
    RUN(8,96,16,256,1,true,6,true,6);
    RUN(9,96,32,256,1,true,6,true,6);
    RUN(10,112,8,256,1,true,7,true,7);
    RUN(11,112,16,256,1,true,7,true,7);
    RUN(12,112,32,128,2,true,7,true,7);
    RUN(13,112,32,256,1,true,7,true,7);
    RUN(14,48,16,512,1,true,3,true,3);
    RUN(15,80,16,512,1,true,5,true,5);
    default:TORCH_CHECK(false,"Unknown TMA four-warp configuration");
  }
#undef RUN
}
}
