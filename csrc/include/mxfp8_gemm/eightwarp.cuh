#pragma once
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
#include "mxfp8_gemm/validation.hpp"
#include "mxfp8_gemm/pdl.cuh"

namespace mxfp8_eightwarp {
constexpr int Warps=8;
using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,cutlass::float_ue8m0_t,32>;

__device__ __forceinline__ int sf_index(int r,int g,int k) {
  return (r/128)*(k/128)*512+(g/4)*512+(r%32)*16+((r%128)/32)*4+g%4;
}
template<int BK>
__device__ __forceinline__ int sm_index(int r,int c) {
  return r*BK+(c^((r%8)*16));
}
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
template<int BM,int BN,int BK>
__device__ void prefetch(uint8_t* sm,const uint8_t* a,const uint8_t* b,
                         const uint8_t* sa,const uint8_t* sb,
                         int pm,int pn,int start,int m,int n,int k) {
#pragma unroll
  for(int i=threadIdx.x;i<(BM+BN)*(BK/16);i+=Warps*32) {
    int row=i/(BK/16),col=(i%(BK/16))*16;
    bool is_a=row<BM;
    int r=is_a?pm+row:pn+row-BM;
    int limit=is_a?m:n;
    bool valid=r<limit && start+col<k;
    const uint8_t* src=(is_a?a:b)+(valid?r*k+start+col:0);
    copy16(sm+sm_index<BK>(row,col),src,valid);
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
template<int BM,int BN,int BK,int Stages,bool Swap,int FixedK=0,int WarpM=2,bool PackedSF=false>
__global__ __launch_bounds__(Warps*32) void gemm(
    const uint8_t* __restrict__ a,const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ sa,const uint8_t* __restrict__ sb,
    __nv_bfloat16* __restrict__ out,int m,int n,int dynamic_k, bool pdl) {
  const int k=FixedK?FixedK:dynamic_k;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  extern __shared__ __align__(16) uint8_t storage[];
  constexpr int StageBytes=(BM+BN)*(BK+BK/32);
  constexpr int TM=BM/(16*WarpM),TN=BN/(8*Warps/WarpM);
  int pm=blockIdx.y*BM,pn=blockIdx.x*BN;
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int wm=(warp%WarpM)*16,wn=(warp/WarpM)*8;
  int row=lane/4,col=(lane%4)*4;
  float accum[TM][TN][4]={};
  mxfp8_runtime::dependent_prologue(pdl);
#pragma unroll
  for(int s=0;s<Stages-1;++s)
    prefetch<BM,BN,BK>(storage+s*StageBytes,a,b,sa,sb,pm,pn,s*BK,m,n,k);
  for(int start=0,it=0;start<k;start+=BK,++it) {
    int current=it%Stages;
    prefetch<BM,BN,BK>(storage+((it+Stages-1)%Stages)*StageBytes,
                      a,b,sa,sb,pm,pn,start+(Stages-1)*BK,m,n,k);
    asm volatile("cp.async.wait_group %0;" :: "n"(Stages-1));
    __syncthreads();
    const uint8_t* sm=storage+current*StageBytes;
    uint32_t as_pack[TM][BK/128],bs_pack[TN][BK/128];
    if constexpr(PackedSF) {
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=wm+i*(16*WarpM)+row+(lane%2)*8;
#pragma unroll
        for(int g=0;g<BK/128;++g)
          as_pack[i][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+BN)*BK+r*(BK/32)+g*4);
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        int r=BM+wn+j*(8*Warps/WarpM)+row;
#pragma unroll
        for(int g=0;g<BK/128;++g)
          bs_pack[j][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+BN)*BK+r*(BK/32)+g*4);
      }
    }
#pragma unroll
    for(int kk=0;kk<BK;kk+=32) {
      uint32_t av[TM][4],bv[TN][2];
      uint8_t as[TM],bs[TN];
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=wm+i*(16*WarpM);
        const auto* ptr=reinterpret_cast<const cute::uint128_t*>(
            sm+sm_index<BK>(r+lane%16,kk+(lane/16)*16));
        cute::SM75_U32x4_LDSM_N::copy(*ptr,av[i][0],av[i][1],av[i][2],av[i][3]);
        as[i]=PackedSF ? uint8_t(as_pack[i][kk/128] >> ((kk/32%4)*8)) : sm[(BM+BN)*BK+(r+row+(lane%2)*8)*(BK/32)+kk/32];
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        int r=wn+j*(8*Warps/WarpM)+row;
        const auto* ptr=reinterpret_cast<const cute::uint128_t*>(
            sm+sm_index<BK>(BM+wn+j*(8*Warps/WarpM)+lane%8,kk+((lane/8)%2)*16));
        cute::SM75_U32x2_LDSM_N::copy(*ptr,bv[j][0],bv[j][1]);
        bs[j]=PackedSF ? uint8_t(bs_pack[j][kk/128] >> ((kk/32%4)*8)) : sm[(BM+BN)*BK+(BM+r)*(BK/32)+kk/32];
      }
#pragma unroll
      for(int i=0;i<TM;++i) {
#pragma unroll
        for(int j=0;j<TN;++j) {
          auto& c=accum[i][j];
          Mma::fma(c[0],c[1],c[2],c[3],av[i][0],av[i][1],av[i][2],av[i][3],
                   bv[j][0],bv[j][1],c[0],c[1],c[2],c[3],as[i],bs[j]);
        }
      }
    }
    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;");
  __syncthreads(); // All asynchronous writes must finish before storage reuse.
  constexpr int Rows=Swap?BN:BM, Cols=Swap?BM:BN, Stride=Cols+8;
  // Shared allocation also covers the padded output tile for single-stage kernels.
  auto* tile=reinterpret_cast<__nv_bfloat16*>(storage);
#pragma unroll
  for(int i=0;i<TM;++i) {
#pragma unroll
    for(int j=0;j<TN;++j) {
#pragma unroll
      for(int v=0;v<4;++v) {
        int r=wm+i*(16*WarpM)+row+(v/2)*8;
        int c=wn+j*(8*Warps/WarpM)+(lane%4)*2+v%2;
        tile[Swap?c*Stride+r:r*Stride+c]=__float2bfloat16_rn(accum[i][j][v]);
      }
    }
  }
  __syncthreads();
#pragma unroll
  for(int i=threadIdx.x;i<Rows*(Cols/8);i+=Warps*32) {
    int r=i/(Cols/8),c=(i%(Cols/8))*8;
    int gr=(Swap?pn:pm)+r,gc=(Swap?pm:pn)+c;
    int out_rows=Swap?n:m, out_cols=Swap?m:n;
    if(gr<out_rows && gc+7<out_cols) {
      *reinterpret_cast<uint4*>(out+int64_t(gr)*out_cols+gc)=
          *reinterpret_cast<const uint4*>(tile+r*Stride+c);
    }
  }
#else
  asm volatile("trap;");
#endif
}

template<int BM,int BN,int BK,int Stages,bool Swap,int FixedK=0,int WarpM=2,bool PackedSF=false>
void launch(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,
            at::Tensor const& sb,at::Tensor const& out) {
  int m=Swap?b.size(0):a.size(0),n=Swap?a.size(0):b.size(0),k=a.size(1);
  constexpr int PipelineBytes=(BM+BN)*(BK+BK/32)*Stages;
  constexpr int OutputBytes=(Swap?BN:BM)*((Swap?BM:BN)+8)*2;
  constexpr int Bytes=PipelineBytes>OutputBytes?PipelineBytes:OutputBytes;
  auto fn=gemm<BM,BN,BK,Stages,Swap,FixedK,WarpM,PackedSF>;
  if constexpr(Bytes>=48*1024) {
    auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes);
    TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
  }
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
  bool pdl=mxfp8_runtime::pdl_enabled();
  auto err=mxfp8_runtime::launch_dependent_kernel(
    pdl,fn,dim3((n+BN-1)/BN,(m+BM-1)/BM),Warps*32,Bytes,stream,
    static_cast<const uint8_t*>((Swap?b:a).data_ptr()),
    static_cast<const uint8_t*>((Swap?a:b).data_ptr()),
    static_cast<const uint8_t*>((Swap?sb:sa).data_ptr()),
    static_cast<const uint8_t*>((Swap?sa:sb).data_ptr()),
    static_cast<__nv_bfloat16*>(out.data_ptr()),m,n,k,pdl);
  TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
}

template<int BM,int BN,int BK,int Stages,bool Swap,int WarpM=2,bool PackedSF=false>
void specialized(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,at::Tensor const& out) {
  if(a.size(1)==2048) launch<BM,BN,BK,Stages,Swap,2048,WarpM,PackedSF>(a,b,sa,sb,out);
  else if(a.size(1)==2560) launch<BM,BN,BK,Stages,Swap,2560,WarpM,PackedSF>(a,b,sa,sb,out);
  else launch<BM,BN,BK,Stages,Swap,0,WarpM,PackedSF>(a,b,sa,sb,out);
}
void mm_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,
            at::Tensor const& sb,at::Tensor const& out,int64_t config) {
  mxfp8_common::validate(a,b,sa,sb,out);
  c10::cuda::CUDAGuard guard(a.device());
#define RUN(ID,...) case ID:specialized<__VA_ARGS__>(a,b,sa,sb,out);break
  switch(config) {
    RUN(0,128,128,128,2,false,4,true);
    RUN(1,64,128,128,3,false,2,true);
    RUN(2,128,64,128,3,false,4,true);
    RUN(3,128,256,128,1,false,4,true);
    RUN(4,256,128,128,1,false,4,true);
    RUN(5,64,64,256,2,false,4,true);
    RUN(6,128,16,128,2,true,8,true);
    RUN(7,128,32,128,2,true,4,true);
    RUN(8,128,64,128,2,true,4,true);
    RUN(9,64,16,256,2,true,4,true);
    RUN(10,64,32,256,1,true,4,true);
    RUN(11,64,16,256,1,true,4,true);
    default:TORCH_CHECK(false,"Unknown specialized configuration");
  }
#undef RUN
}
}
