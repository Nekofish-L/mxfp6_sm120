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
#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/arch/barrier.h"
#include "mxfp8_gemm/validation.hpp"

namespace mxfp8_tma_scales {
using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,cutlass::float_ue8m0_t,32>;

__device__ __forceinline__ int sf_index(int r,int g,int k) {
  return (r/128)*(k/128)*512+(g/4)*512+(r%32)*16+((r%128)/32)*4+g%4;
}
template<int BK,int Rows>
__device__ __forceinline__ int sm_index(int r,int c) {
  return (c/128)*Rows*128+r*128+((c%128)^((r%8)*16));
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
                         int pm,int pn,int start,int m,int n,int k, CUtensorMap const* ma, CUtensorMap const* mb, CUtensorMap const* msa, CUtensorMap const* msb, uint64_t* barrier) {
  static_assert(BK%128==0 && 128%BM==0 && 128%BN==0);
  if(threadIdx.x==0) {
    cute::set_barrier_transaction_bytes(*barrier,(BM+BN)*BK+2*(BK/128)*512);
    cute::SM90_TMA_LOAD_3D::copy(ma,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm,0,pm,start/128);
    cute::SM90_TMA_LOAD_3D::copy(mb,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm+BM*BK,0,pn,start/128);
  }
  if(threadIdx.x==0) {
    cute::SM90_TMA_LOAD_3D::copy(msa,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm+(BM+BN)*BK,0,start/32,pm/128);
    cute::SM90_TMA_LOAD_3D::copy(msb,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm+(BM+BN)*BK+(BK/128)*512,0,start/32,pn/128);
  }

}
template<int BM,int BN,int BK,int Stages,bool Swap,int FixedK=0,int WarpM=2,bool PackedSF=false>
__global__ __launch_bounds__(128) void gemm(
    const uint8_t* __restrict__ a,const uint8_t* __restrict__ b,
    const uint8_t* __restrict__ sa,const uint8_t* __restrict__ sb,
    __nv_bfloat16* __restrict__ out,int m,int n,int dynamic_k, const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, const __grid_constant__ CUtensorMap msa, const __grid_constant__ CUtensorMap msb) {
  const int k=FixedK?FixedK:dynamic_k;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  extern __shared__ __align__(1024) uint8_t storage[];
  __shared__ uint64_t barriers[Stages];
  if(threadIdx.x==0) {
    for(int i=0;i<Stages;++i) cute::initialize_barrier(barriers[i]);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  constexpr int StageBytes=((BM+BN)*BK+2*(BK/128)*512+1023)/1024*1024;
  constexpr int TM=BM/(16*WarpM),TN=BN/(32/WarpM);
  int pm=blockIdx.y*BM,pn=blockIdx.x*BN;
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int wm=(warp%WarpM)*16,wn=(warp/WarpM)*8;
  int row=lane/4,col=(lane%4)*4;
  float accum[TM][TN][4]={};
#pragma unroll
  for(int s=0;s<Stages-1;++s)
    prefetch<BM,BN,BK>(storage+s*StageBytes,a,b,sa,sb,pm,pn,s*BK,m,n,k,&ma,&mb,&msa,&msb,barriers+s);
  for(int start=0,it=0;start<k;start+=BK,++it) {
    int current=it%Stages;
    prefetch<BM,BN,BK>(storage+((it+Stages-1)%Stages)*StageBytes,
                      a,b,sa,sb,pm,pn,start+(Stages-1)*BK,m,n,k,&ma,&mb,&msa,&msb,barriers+((it+Stages-1)%Stages));
    cute::wait_barrier(barriers[current],(it/Stages)&1);
    __syncthreads();
    const uint8_t* sm=storage+current*StageBytes;
    uint32_t as_pack[TM][BK/128],bs_pack[TN][BK/128];
    if constexpr(PackedSF) {
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=pm%128+wm+i*(16*WarpM)+row+(lane%2)*8;
#pragma unroll
        for(int g=0;g<BK/128;++g)
          as_pack[i][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+BN)*BK+g*512+(r%32)*16+(r/32)*4);
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        int r=pn%128+wn+j*(32/WarpM)+row;
#pragma unroll
        for(int g=0;g<BK/128;++g)
          bs_pack[j][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+BN)*BK+(BK/128)*512+g*512+(r%32)*16+(r/32)*4);
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
            sm+sm_index<BK,BM>(r+lane%16,kk+(lane/16)*16));
        cute::SM75_U32x4_LDSM_N::copy(*ptr,av[i][0],av[i][1],av[i][2],av[i][3]);
        as[i]=uint8_t(as_pack[i][kk/128] >> ((kk/32%4)*8));
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        int r=wn+j*(32/WarpM)+row;
        const auto* ptr=reinterpret_cast<const cute::uint128_t*>(
            sm+BM*BK+sm_index<BK,BN>(wn+j*(32/WarpM)+lane%8,kk+((lane/8)%2)*16));
        cute::SM75_U32x2_LDSM_N::copy(*ptr,bv[j][0],bv[j][1]);
        bs[j]=uint8_t(bs_pack[j][kk/128] >> ((kk/32%4)*8));
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
  // Drain every prefetched TMA before reusing its pipeline storage.
  for(int s=0;s<Stages;++s) {
    int last=(k+BK-1)/BK+Stages-2;
    if(s<=last) cute::wait_barrier(barriers[s],((last-s)/Stages)&1);
  }
  __syncthreads();
  constexpr int Rows=Swap?BN:BM, Cols=Swap?BM:BN, Stride=Cols+8;
  static_assert(Rows*Stride*2 <= StageBytes*Stages, "Output tile must fit the reused pipeline storage");
  auto* tile=reinterpret_cast<__nv_bfloat16*>(storage);
#pragma unroll
  for(int i=0;i<TM;++i) {
#pragma unroll
    for(int j=0;j<TN;++j) {
#pragma unroll
      for(int v=0;v<4;++v) {
        int r=wm+i*(16*WarpM)+row+(v/2)*8;
        int c=wn+j*(32/WarpM)+(lane%4)*2+v%2;
        tile[Swap?c*Stride+r:r*Stride+c]=__float2bfloat16_rn(accum[i][j][v]);
      }
    }
  }
  __syncthreads();
#pragma unroll
  for(int i=threadIdx.x;i<Rows*(Cols/8);i+=128) {
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
  constexpr int Bytes=(((BM+BN)*BK+2*(BK/128)*512+1023)/1024*1024)*Stages;
  auto fn=gemm<BM,BN,BK,Stages,Swap,FixedK,WarpM,PackedSF>;
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
  auto scale_descriptor=[&](void* ptr,int rows) {
    CUtensorMap map{};
    uint64_t dimensions[3]={128,uint64_t(k/32),uint64_t((rows+127)/128)},strides[2]={128,uint64_t(k)*4};
    uint32_t box[3]={128,BK/32,1},element_strides[3]={1,1,1};
    CUresult status=CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,ptr,dimensions,strides,box,element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"Scale TMA descriptor failed: ",int(status));
    return map;
  };
  auto msa=scale_descriptor((Swap?sb:sa).data_ptr(),m);
  auto msb=scale_descriptor((Swap?sa:sb).data_ptr(),n);
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
  fn<<<dim3((n+BN-1)/BN,(m+BM-1)/BM),128,Bytes,stream>>>(
    static_cast<const uint8_t*>((Swap?b:a).data_ptr()),
    static_cast<const uint8_t*>((Swap?a:b).data_ptr()),
    static_cast<const uint8_t*>((Swap?sb:sa).data_ptr()),
    static_cast<const uint8_t*>((Swap?sa:sb).data_ptr()),
    static_cast<__nv_bfloat16*>(out.data_ptr()),m,n,k,ma,mb,msa,msb);
  auto err=cudaGetLastError();
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
    RUN(0,64,16,256,2,true,2,true);
    RUN(1,64,32,256,1,true,2,true);
    RUN(2,64,32,256,2,true,2,true);
    RUN(3,64,64,256,1,true,2,true);
    RUN(4,64,64,256,2,true,2,true);
    RUN(5,128,16,256,1,true,4,true);
    RUN(6,128,16,256,2,true,4,true);
    RUN(7,128,32,256,1,true,2,true);
    RUN(8,128,32,256,2,true,2,true);
    RUN(9,32,32,256,2,true,2,true);
    RUN(10,64,64,256,2,false,2,true);
    RUN(11,128,16,128,3,true,4,true);
    default:TORCH_CHECK(false,"Unknown TMA four-warp configuration");
  }
#undef RUN
}
}
