#pragma once
#include "mxfp8_gemm/layout.cuh"
// Diagnostic alternative to the 384-thread CUTLASS warp-specialized templates.
// Uses CUTLASS/CuTe's SM120 block-scaled MMA atom, four warps, cp.async,
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
#include "mxfp8_gemm/pdl.cuh"

namespace mxfp8_tma_eightwarp {
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
    __nv_bfloat16* __restrict__ out,int m,int n,int dynamic_k, bool pdl, const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb) {
  constexpr bool ReserveOutput=true;
#include "mxfp8_gemm/kernels/tma_warp_body.inc"
}

template<int BM,int BN,int BK,int Stages,bool Swap,int FixedK=0,int WarpM=2,bool PackedSF=false,int Warps=4>
void launch(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,
            at::Tensor const& sb,at::Tensor const& out) {
  int m=Swap?b.size(0):a.size(0),n=Swap?a.size(0):b.size(0),k=a.size(1);
  constexpr int PipelineBytes=(((BM+BN)*(BK+BK/32)+1023)/1024*1024)*Stages;
  constexpr int OutputBytes=(Swap?BN:BM)*((Swap?BM:BN)+8)*2;
  constexpr int Bytes=PipelineBytes>OutputBytes?PipelineBytes:OutputBytes;
  TORCH_CHECK(Bytes + Stages*sizeof(uint64_t) <= at::cuda::getDeviceProperties(a.get_device())->sharedMemPerBlockOptin,
              "TMA eight-warp tile exceeds hardware shared memory capacity");
  auto fn=gemm<BM,BN,BK,Stages,Swap,FixedK,WarpM,PackedSF,Warps>;
  if constexpr(Bytes>=48*1024) {
    auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes);
    TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
  }
  auto descriptor=[&](void* ptr,int rows,int tile_rows) {
    CUtensorMap map{};
    uint64_t dimensions[3]={128,uint64_t(rows),uint64_t(k/128)}, strides[2]={uint64_t(k),128};
    uint32_t box[3]={128,uint32_t(tile_rows),BK/128}, element_strides[3]={1,1,1};
    CUresult status=CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,ptr,dimensions,strides,box,element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"TMA descriptor failed: ",int(status));
    return map;
  };
  auto ma=descriptor((Swap?b:a).data_ptr(),m,BM);
  auto mb=descriptor((Swap?a:b).data_ptr(),n,BN);
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
  bool pdl=mxfp8_runtime::pdl_enabled();
  auto err=mxfp8_runtime::launch_dependent_kernel(
    pdl,fn,dim3((n+BN-1)/BN,(m+BM-1)/BM),Warps*32,Bytes,stream,
    static_cast<const uint8_t*>((Swap?b:a).data_ptr()),
    static_cast<const uint8_t*>((Swap?a:b).data_ptr()),
    static_cast<const uint8_t*>((Swap?sb:sa).data_ptr()),
    static_cast<const uint8_t*>((Swap?sa:sb).data_ptr()),
    static_cast<__nv_bfloat16*>(out.data_ptr()),m,n,k,pdl,ma,mb);
  TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
}

template<int BM,int BN,int BK,int Stages,bool Swap,int WarpM=2,bool PackedSF=false,int Warps=4>
void specialized(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,at::Tensor const& out) {
  if(a.size(1)==2048) launch<BM,BN,BK,Stages,Swap,2048,WarpM,PackedSF,Warps>(a,b,sa,sb,out);
  else if(a.size(1)==2560) launch<BM,BN,BK,Stages,Swap,2560,WarpM,PackedSF,Warps>(a,b,sa,sb,out);
  else launch<BM,BN,BK,Stages,Swap,0,WarpM,PackedSF,Warps>(a,b,sa,sb,out);
}
void mm_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,
            at::Tensor const& sb,at::Tensor const& out,int64_t config) {
  mxfp8_common::validate(a,b,sa,sb,out);
  c10::cuda::CUDAGuard guard(a.device());
#define RUN(ID,...) case ID:specialized<__VA_ARGS__>(a,b,sa,sb,out);break
  switch(config) {
    RUN(0,128,128,128,2,false,4,true,8);
    RUN(1,64,128,128,3,false,2,true,8);
    RUN(2,128,64,128,3,false,4,true,8);
    RUN(3,128,256,128,1,false,4,true,8);
    RUN(4,256,128,128,1,false,4,true,8);
    RUN(5,64,64,256,2,false,4,true,8);
    RUN(6,128,16,128,2,true,8,true,8);
    RUN(7,128,32,128,2,true,4,true,8);
    RUN(8,128,64,128,2,true,4,true,8);
    RUN(9,64,16,256,2,true,4,true,8);
    RUN(10,64,32,256,1,true,4,true,8);
    RUN(11,64,16,256,1,true,4,true,8);
    default:TORCH_CHECK(false,"Unknown TMA four-warp configuration");
  }
#undef RUN
}
}
