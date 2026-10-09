// SPDX-License-Identifier: BSD-3-Clause
// Derived from mxfp6 cfbc5074 csrc/include/mxfp8_gemm/tma256.cuh
// SHA256 4af70650622482c7c8422037b9e71ebfb5b3ad0f533bf3151ee795544b42eb00.
// QKVZ/MLP batches 33–128; original M64 QKVZ cfg2 arithmetic is unchanged.
// Two TMA value stages; packed E8M0 SF is loaded directly into registers.
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include "cute/arch/mma_sm120.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/arch/barrier.h"
#include "mxfp8_gemm/dual.cuh"

namespace mxfp8_dual_pipereg {
using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,cutlass::float_ue8m0_t,32>;

using mxfp8_dual_detail::sf_index;
using mxfp8_dual_detail::sm_index;

template<int BM,int BN,int BK>
__device__ void prefetch(uint8_t* sm,int pm,int pn,int start,
                         CUtensorMap const* mw,CUtensorMap const* mhi,CUtensorMap const* mres,
                         uint64_t* barrier) {
  static_assert(BK%128==0);
  if(threadIdx.x==0) {
    cute::set_barrier_transaction_bytes(*barrier,(BM+2*BN)*BK);
    cute::SM90_TMA_LOAD_3D::copy(mw,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm,0,pm,start/128);
    cute::SM90_TMA_LOAD_3D::copy(mhi,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm+BM*BK,0,pn,start/128);
    cute::SM90_TMA_LOAD_3D::copy(mres,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm+(BM+BN)*BK,0,pn,start/128);
  }
}
template<int BM,int BN,int BK,int Stages,int FixedK,int WarpM,int FixedN>
__global__ __launch_bounds__(128) void gemm(
    const uint8_t* __restrict__ sw,const uint8_t* __restrict__ shi,const uint8_t* __restrict__ sres,
    __nv_bfloat16* __restrict__ out,int n,
    const __grid_constant__ CUtensorMap mw,const __grid_constant__ CUtensorMap mhi,
    const __grid_constant__ CUtensorMap mres,bool pdl) {
  constexpr int m=FixedN,k=FixedK;
  static_assert(m%BM==0 && k%BK==0,
                "Weight scale packs require full valid N/K tiles");
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  extern __shared__ __align__(1024) uint8_t storage[];
  __shared__ uint64_t barriers[Stages];
  if(threadIdx.x==0) {
    for(int i=0;i<Stages;++i) cute::initialize_barrier(barriers[i]);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  constexpr int StageBytes=(BM+2*BN)*BK;
  constexpr int TM=BM/(16*WarpM),TN=BN/(32/WarpM);
  int pm=blockIdx.y*BM,pn=blockIdx.x*BN;
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int wm=(warp%WarpM)*16,wn=(warp/WarpM)*8;
  int row=lane/4;
  float accum_hi[TM][TN][4]={};
  float accum_res[TM][TN][4]={};
  mxfp_common::dependent_prologue(pdl);
#pragma unroll
  for(int s=0;s<Stages-1;++s)
    prefetch<BM,BN,BK>(storage+s*StageBytes,pm,pn,s*BK,&mw,&mhi,&mres,barriers+s);
  for(int start=0,it=0;start<k;start+=BK,++it) {
    int current=it%Stages;
    prefetch<BM,BN,BK>(storage+((it+Stages-1)%Stages)*StageBytes,
                      pm,pn,start+(Stages-1)*BK,&mw,&mhi,&mres,barriers+((it+Stages-1)%Stages));
    // These global scale loads are independent of the current TMA value slab.
    // Keep each packed four-E8M0 group in registers while the TMA barrier waits.
    uint32_t as_pack[TM][BK/128],bs_hi_pack[TN][BK/128],bs_res_pack[TN][BK/128];
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=wm+i*(16*WarpM)+row+(lane%2)*8;
#pragma unroll
        for(int g=0;g<BK/128;++g)
          as_pack[i][g]=*reinterpret_cast<const uint32_t*>(
              sw+sf_index(pm+r,start/32+g*4,k));
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        int r=wn+j*(32/WarpM)+row;
#pragma unroll
        for(int g=0;g<BK/128;++g) {
          // Invalid activation rows are TMA zero-filled; never read their scales.
          bs_hi_pack[j][g]=pn+r<n?*reinterpret_cast<const uint32_t*>(
              shi+sf_index(pn+r,start/32+g*4,k)):0;
          bs_res_pack[j][g]=pn+r<n?*reinterpret_cast<const uint32_t*>(
              sres+sf_index(pn+r,start/32+g*4,k)):0;
        }
      }
    cute::wait_barrier(barriers[current],(it/Stages)&1);
    __syncthreads();
    const uint8_t* sm=storage+current*StageBytes;
#pragma unroll
    for(int kk=0;kk<BK;kk+=32) {
      uint32_t av[TM][4],bv_hi[TN][2],bv_res[TN][2];
      uint8_t as[TM],bs_hi[TN],bs_res[TN];
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=wm+i*(16*WarpM);
        const auto* ptr=reinterpret_cast<const cute::uint128_t*>(
            sm+sm_index<BM>(r+lane%16,kk+(lane/16)*16));
        cute::SM75_U32x4_LDSM_N::copy(*ptr,av[i][0],av[i][1],av[i][2],av[i][3]);
        as[i]=uint8_t(as_pack[i][kk/128] >> ((kk/32%4)*8));
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        const auto* ptr_hi=reinterpret_cast<const cute::uint128_t*>(
            sm+BM*BK+sm_index<BN>(wn+j*(32/WarpM)+lane%8,kk+((lane/8)%2)*16));
        const auto* ptr_res=reinterpret_cast<const cute::uint128_t*>(
            sm+(BM+BN)*BK+sm_index<BN>(wn+j*(32/WarpM)+lane%8,kk+((lane/8)%2)*16));
        cute::SM75_U32x2_LDSM_N::copy(*ptr_hi,bv_hi[j][0],bv_hi[j][1]);
        cute::SM75_U32x2_LDSM_N::copy(*ptr_res,bv_res[j][0],bv_res[j][1]);
        bs_hi[j]=uint8_t(bs_hi_pack[j][kk/128] >> ((kk/32%4)*8));
        bs_res[j]=uint8_t(bs_res_pack[j][kk/128] >> ((kk/32%4)*8));
      }
#pragma unroll
      for(int i=0;i<TM;++i) {
#pragma unroll
        for(int j=0;j<TN;++j) {
          auto& hi=accum_hi[i][j];
          auto& res=accum_res[i][j];
          Mma::fma(hi[0],hi[1],hi[2],hi[3],av[i][0],av[i][1],av[i][2],av[i][3],
                   bv_hi[j][0],bv_hi[j][1],hi[0],hi[1],hi[2],hi[3],as[i],bs_hi[j]);
          Mma::fma(res[0],res[1],res[2],res[3],av[i][0],av[i][1],av[i][2],av[i][3],
                   bv_res[j][0],bv_res[j][1],res[0],res[1],res[2],res[3],as[i],bs_res[j]);
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
  constexpr int Rows=BN, Cols=BM, Stride=Cols+8;
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
        tile[c*Stride+r]=__float2bfloat16_rn(accum_hi[i][j][v]+accum_res[i][j][v]);
      }
    }
  }
  __syncthreads();
#pragma unroll
  for(int i=threadIdx.x;i<Rows*(Cols/8);i+=128) {
    int r=i/(Cols/8),c=(i%(Cols/8))*8;
    int gr=pn+r,gc=pm+c;
    int out_rows=n,out_cols=m;
    if(gr<out_rows && gc+7<out_cols) {
      *reinterpret_cast<uint4*>(out+int64_t(gr)*out_cols+gc)=
          *reinterpret_cast<const uint4*>(tile+r*Stride+c);
    }
  }
#else
  asm volatile("trap;");
#endif
}

template<int N>
void launch(at::Tensor const& hi,at::Tensor const& weight,
            at::Tensor const& s_hi,at::Tensor const& s_weight,
            at::Tensor const& res,at::Tensor const& s_res,
            at::Tensor const& out,bool use_pdl) {
  constexpr int K=2560,BM=128,BN=32,BK=128,Stages=2,WarpM=4;
  int const M=int(hi.size(0));
  constexpr int Bytes=(BM+2*BN)*BK*Stages;
  static_assert(Bytes==49152);
  auto fn=gemm<BM,BN,BK,Stages,K,WarpM,N>;
  auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes);
  TORCH_CHECK(err==cudaSuccess,"cudaFuncSetAttribute dynamic shared failed: ",cudaGetErrorString(err));
  auto mw=mxfp8_dual_detail::tensor_map(weight.data_ptr(),N,BM,K,BK);
  auto mhi=mxfp8_dual_detail::tensor_map(hi.data_ptr(),M,BN,K,BK);
  auto mres=mxfp8_dual_detail::tensor_map(res.data_ptr(),M,BN,K,BK);
  auto stream=c10::cuda::getCurrentCUDAStream(hi.get_device());
  auto error=mxfp_common::launch_dependent_kernel(
    use_pdl,fn,dim3((M+BN-1)/BN,(N+BM-1)/BM),128,Bytes,stream,
    static_cast<const uint8_t*>(s_weight.data_ptr()),
    static_cast<const uint8_t*>(s_hi.data_ptr()),
    static_cast<const uint8_t*>(s_res.data_ptr()),
    static_cast<__nv_bfloat16*>(out.data_ptr()),M,mw,mhi,mres,use_pdl);
  TORCH_CHECK(error==cudaSuccess,"kernel launch failed: ",cudaGetErrorString(error));
}

void gemm_out(at::Tensor const& hi,at::Tensor const& weight,
              at::Tensor const& s_hi,at::Tensor const& s_weight,
              at::Tensor const& res,at::Tensor const& s_res,
              at::Tensor const& out,bool use_pdl) {
  if (weight.size(0)==12288) launch<12288>(hi,weight,s_hi,s_weight,res,s_res,out,use_pdl);
  else launch<18432>(hi,weight,s_hi,s_weight,res,s_res,out,use_pdl);
}
} // namespace mxfp8_dual_pipereg
