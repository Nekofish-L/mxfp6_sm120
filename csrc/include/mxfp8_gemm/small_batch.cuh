#pragma once
#include "mxfp8_gemm/pdl.cuh"
// Unified decode kernel: TMA values, cp.async scales, register-sourced CuTe MMA.
// BM indexes weight rows, BN indexes activation rows (operands are transposed).
// WarpK partitions each K slab inside the CTA; partial sums remain FP32 until
// the epilogue. All shapes use the same pipeline and existing packed SF layout.
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

namespace mxfp8_small_batch {
using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,cutlass::float_ue8m0_t,32>;

__device__ __forceinline__ int sf_index(int r,int g,int k) {
  return (r/128)*(k/128)*512+(g/4)*512+(r%32)*16+((r%128)/32)*4+g%4;
}
template<int BK,int Rows>
__device__ __forceinline__ int sm_index(int r,int c) {
  return (c/128)*Rows*128+r*128+((c%128)^((r%8)*16));
}
__device__ __forceinline__ void copy4(uint8_t* dst,const uint8_t* src,bool valid) {
  uint32_t addr=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
               "r"(addr),"l"(src),"r"(valid?4:0));
}
template<int BM,int BN,int BK,int Warps>
__device__ void prefetch(uint8_t* sm,const uint8_t* sa,const uint8_t* sb,
                         int pm,int pn,int start,int m,int n,int k, CUtensorMap const* ma, CUtensorMap const* mb, uint64_t* barrier) {
  static_assert(BK%128==0);
  if(threadIdx.x==0 && start<k) {
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
template<int BM,int BN,int BK,int Stages,int WarpM,int Warps,int WarpK=1>
__global__ __launch_bounds__(Warps*32) void gemm(
    const uint8_t* __restrict__ sa,const uint8_t* __restrict__ sb,
    __nv_bfloat16* __restrict__ out,int m,int n,int k, const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mb, bool pdl) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  extern __shared__ __align__(1024) uint8_t storage[];
  __shared__ uint64_t barriers[Stages];
  if(threadIdx.x==0) {
    for(int i=0;i<Stages;++i) cute::initialize_barrier(barriers[i]);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  constexpr int StageBytes=((BM+BN)*(BK+BK/32)+1023)/1024*1024;
  constexpr int OutputWarps=Warps/WarpK, KPerWarp=BK/WarpK;
  static_assert(KPerWarp%128==0);
  constexpr int TM=BM/(16*WarpM),TN=BN/(8*OutputWarps/WarpM);
  int pm=blockIdx.y*BM,pn=blockIdx.x*BN;
  int lane=threadIdx.x%32,warp=(threadIdx.x/32)%OutputWarps;
  int wk=threadIdx.x/(32*OutputWarps);
  int wm=(warp%WarpM)*16,wn=(warp/WarpM)*8;
  int row=lane/4;
  float accum[TM][TN][4]={};
  mxfp8_runtime::dependent_prologue(pdl);
#pragma unroll
  for(int s=0;s<Stages-1;++s)
    prefetch<BM,BN,BK,Warps>(storage+s*StageBytes,sa,sb,pm,pn,s*BK,m,n,k,&ma,&mb,barriers+s);
  for(int start=0,it=0;start<k;start+=BK,++it) {
    int current=it%Stages;
    prefetch<BM,BN,BK,Warps>(storage+((it+Stages-1)%Stages)*StageBytes,
                      sa,sb,pm,pn,start+(Stages-1)*BK,m,n,k,&ma,&mb,barriers+((it+Stages-1)%Stages));
    asm volatile("cp.async.wait_group %0;" :: "n"(Stages-1));
    cute::wait_barrier(barriers[current],(it/Stages)&1);
    __syncthreads();
    const uint8_t* sm=storage+current*StageBytes;
    uint32_t as_pack[TM][KPerWarp/128],bs_pack[TN][KPerWarp/128];
    {
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=wm+i*(16*WarpM)+row+(lane%2)*8;
#pragma unroll
        for(int g=0;g<KPerWarp/128;++g)
          as_pack[i][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+BN)*BK+r*(BK/32)+(g+wk*(KPerWarp/128))*4);
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        int r=BM+wn+j*(8*OutputWarps/WarpM)+row;
#pragma unroll
        for(int g=0;g<KPerWarp/128;++g)
          bs_pack[j][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+BN)*BK+r*(BK/32)+(g+wk*(KPerWarp/128))*4);
      }
    }
#pragma unroll
    for(int ki=0;ki<KPerWarp;ki+=32) {
      int kk=ki+wk*KPerWarp;
      uint32_t av[TM][4],bv[TN][2];
      uint8_t as[TM],bs[TN];
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=wm+i*(16*WarpM);
        const auto* ptr=reinterpret_cast<const cute::uint128_t*>(
            sm+sm_index<BK,BM>(r+lane%16,kk+(lane/16)*16));
        cute::SM75_U32x4_LDSM_N::copy(*ptr,av[i][0],av[i][1],av[i][2],av[i][3]);
        as[i]=uint8_t(as_pack[i][ki/128] >> ((ki/32%4)*8));
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        const auto* ptr=reinterpret_cast<const cute::uint128_t*>(
            sm+BM*BK+sm_index<BK,BN>(wn+j*(8*OutputWarps/WarpM)+lane%8,kk+((lane/8)%2)*16));
        cute::SM75_U32x2_LDSM_N::copy(*ptr,bv[j][0],bv[j][1]);
        bs[j]=uint8_t(bs_pack[j][ki/128] >> ((ki/32%4)*8));
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
  // Only valid TMA slabs are issued. Every slab has been waited on above;
  // drain the zero-fill cp.async tail before reusing shared storage.
  __syncthreads();
  constexpr int Rows=BN, Cols=BM, Stride=Cols+8;
  static_assert(Rows*Stride*2 <= StageBytes*Stages, "Output tile must fit the reused pipeline storage");
  if constexpr(WarpK==1) {
    auto* tile=reinterpret_cast<__nv_bfloat16*>(storage);
#pragma unroll
    for(int i=0;i<TM;++i) {
#pragma unroll
      for(int j=0;j<TN;++j) {
#pragma unroll
        for(int v=0;v<4;++v) {
          int r=wm+i*(16*WarpM)+row+(v/2)*8;
          int c=wn+j*(8*OutputWarps/WarpM)+(lane%4)*2+v%2;
          tile[c*Stride+r]=__float2bfloat16_rn(accum[i][j][v]);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for(int i=threadIdx.x;i<Rows*(Cols/8);i+=Warps*32) {
      int r=i/(Cols/8),c=(i%(Cols/8))*8;
      int gr=pn+r,gc=pm+c;
      int out_rows=n, out_cols=m;
      if(gr<out_rows && gc+7<out_cols) {
        *reinterpret_cast<uint4*>(out+int64_t(gr)*out_cols+gc)=
            *reinterpret_cast<const uint4*>(tile+r*Stride+c);
      }
    }
  } else {
    static_assert(WarpK*Rows*Stride*4 <= StageBytes*Stages);
    auto* partial=reinterpret_cast<float*>(storage);
#pragma unroll
    for(int i=0;i<TM;++i) {
#pragma unroll
      for(int j=0;j<TN;++j) {
#pragma unroll
        for(int v=0;v<4;++v) {
          int r=wm+i*(16*WarpM)+row+(v/2)*8;
          int c=wn+j*(8*OutputWarps/WarpM)+(lane%4)*2+v%2;
          partial[wk*Rows*Stride+c*Stride+r]=accum[i][j][v];
        }
      }
    }
    __syncthreads();
    for(int i=threadIdx.x;i<Rows*(Cols/8);i+=Warps*32) {
      int r=i/(Cols/8),c=(i%(Cols/8))*8;
      if(pn+r<n && pm+c+7<m) {
        alignas(16) __nv_bfloat16 values[8];
#pragma unroll
        for(int v=0;v<8;++v) {
          float sum=0;
#pragma unroll
          for(int split=0;split<WarpK;++split)
            sum+=partial[split*Rows*Stride+r*Stride+c+v];
          values[v]=__float2bfloat16_rn(sum);
        }
        *reinterpret_cast<uint4*>(out+int64_t(pn+r)*m+pm+c)=
            *reinterpret_cast<const uint4*>(values);
      }
    }
  }
#else
  asm volatile("trap;");
#endif
}

template<int BM,int BN,int BK,int Stages,int WarpM,int Warps,int WarpK=1>
void launch(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,
            at::Tensor const& sb,at::Tensor const& out) {
  int m=b.size(0),n=a.size(0),k=a.size(1);
  constexpr int Bytes=(((BM+BN)*(BK+BK/32)+1023)/1024*1024)*Stages;
  auto fn=gemm<BM,BN,BK,Stages,WarpM,Warps,WarpK>;
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
  auto ma=descriptor(b.data_ptr(),m,BM);
  auto mb=descriptor(a.data_ptr(),n,BN);
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
  bool pdl=mxfp8_runtime::pdl_enabled();
  auto err=mxfp8_runtime::launch_dependent_kernel(
    pdl,fn,dim3((n+BN-1)/BN,(m+BM-1)/BM),Warps*32,Bytes,stream,

    static_cast<const uint8_t*>(sb.data_ptr()),
    static_cast<const uint8_t*>(sa.data_ptr()),
    static_cast<__nv_bfloat16*>(out.data_ptr()),m,n,k,ma,mb,pdl);
  TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
}

// One runtime-K implementation; tile and pipeline parameters only.
void mm_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,
            at::Tensor const& sb,at::Tensor const& out,int64_t config) {
  mxfp8_common::validate(a,b,sa,sb,out);
  c10::cuda::CUDAGuard guard(a.device());
#define RUN(ID,BM,BN,BK,S,WM,W,WK) case ID:launch<BM,BN,BK,S,WM,W,WK>(a,b,sa,sb,out);break
  switch(config) {
    RUN(0,16,8,512,3,1,4,4);
    RUN(1,64,16,512,2,2,4,1);
    RUN(2,64,8,256,2,2,2,1);
    RUN(3,128,16,256,2,4,4,1);
    RUN(4,128,8,256,2,4,4,1);
    RUN(5,128,16,128,3,8,8,1);
    RUN(6,128,16,128,2,4,4,1);
    RUN(7,64,16,256,1,2,4,1);
    RUN(8,128,16,256,1,4,4,1);
    RUN(9,32,16,512,3,1,4,2);
    RUN(10,64,32,256,3,2,4,1);
    // Compact output widths need more CTAs and CTA-local K parallelism.
    RUN(11,32,32,512,2,1,4,2);
    // Short K: wider activation tiles reuse each weight load across batches.
    RUN(12,128,32,256,2,4,4,1);
    RUN(13,64,64,128,5,2,4,1);
    RUN(14,128,32,128,4,4,4,1);
    default:TORCH_CHECK(false,"Unknown small-batch configuration");
  }
#undef RUN
}
} // namespace mxfp8_small_batch
