// SPDX-License-Identifier: BSD-3-Clause
// Derived from mxfp6 cfbc5074 csrc/include/mxfp8_gemm/tma256.cuh
// SHA256 4af70650622482c7c8422037b9e71ebfb5b3ad0f533bf3151ee795544b42eb00.
// Fixed tactic 665: 128x16x256, one stage, swapped A/B, WarpM=4, packed SF.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "cute/arch/mma_sm120.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/arch/barrier.h"

namespace mxfp8_dual_qkvz32 {
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
__device__ void prefetch(uint8_t* sm,
                         const uint8_t* sw,const uint8_t* shi,const uint8_t* sres,
                         int pm,int pn,int start,int m,int n,int k,
                         CUtensorMap const* mw,CUtensorMap const* mhi,CUtensorMap const* mres,
                         uint64_t* barrier) {
  static_assert(BK%128==0);
  if(threadIdx.x==0) {
    cute::set_barrier_transaction_bytes(*barrier,(BM+2*BN)*BK);
    cute::SM90_TMA_LOAD_3D::copy(mw,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm,0,pm,start/128);
    cute::SM90_TMA_LOAD_3D::copy(mhi,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm+BM*BK,0,pn,start/128);
    cute::SM90_TMA_LOAD_3D::copy(mres,barrier,uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL),sm+(BM+BN)*BK,0,pn,start/128);
  }
#pragma unroll
  for(int i=threadIdx.x;i<(BM+2*BN)*(BK/128);i+=128) {
    int row=i/(BK/128),g=(i%(BK/128))*4;
    bool is_w=row<BM;
    int r=is_w?pm+row:pn+(row-BM)%BN;
    bool valid=r<(is_w?m:n) && start+g*32<k;
    const uint8_t* base=is_w?sw:(row<BM+BN?shi:sres);
    const uint8_t* src=base+(valid?sf_index(r,start/32+g,k):0);
    copy4(sm+(BM+2*BN)*BK+row*(BK/32)+g,src,valid);
  }
  asm volatile("cp.async.commit_group;");
}
template<int BM,int BN,int BK,int Stages,int FixedK,int WarpM>
__global__ __launch_bounds__(128) void gemm(
    const uint8_t* __restrict__ sw,const uint8_t* __restrict__ shi,const uint8_t* __restrict__ sres,
    __nv_bfloat16* __restrict__ out,
    const __grid_constant__ CUtensorMap mw,const __grid_constant__ CUtensorMap mhi,
    const __grid_constant__ CUtensorMap mres) {
  constexpr int m=12288,n=32,k=FixedK;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  extern __shared__ __align__(1024) uint8_t storage[];
  __shared__ uint64_t barriers[Stages];
  if(threadIdx.x==0) {
    for(int i=0;i<Stages;++i) cute::initialize_barrier(barriers[i]);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  constexpr int StageBytes=((BM+2*BN)*(BK+BK/32)+1023)/1024*1024;
  constexpr int TM=BM/(16*WarpM),TN=BN/(32/WarpM);
  int pm=blockIdx.y*BM,pn=blockIdx.x*BN;
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int wm=(warp%WarpM)*16,wn=(warp/WarpM)*8;
  int row=lane/4;
  float accum_hi[TM][TN][4]={};
  float accum_res[TM][TN][4]={};
#pragma unroll
  for(int s=0;s<Stages-1;++s)
    prefetch<BM,BN,BK>(storage+s*StageBytes,sw,shi,sres,pm,pn,s*BK,m,n,k,&mw,&mhi,&mres,barriers+s);
  for(int start=0,it=0;start<k;start+=BK,++it) {
    int current=it%Stages;
    prefetch<BM,BN,BK>(storage+((it+Stages-1)%Stages)*StageBytes,
                      sw,shi,sres,pm,pn,start+(Stages-1)*BK,m,n,k,&mw,&mhi,&mres,barriers+((it+Stages-1)%Stages));
    asm volatile("cp.async.wait_group %0;" :: "n"(Stages-1));
    cute::wait_barrier(barriers[current],(it/Stages)&1);
    __syncthreads();
    const uint8_t* sm=storage+current*StageBytes;
    uint32_t as_pack[TM][BK/128],bs_hi_pack[TN][BK/128],bs_res_pack[TN][BK/128];
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=wm+i*(16*WarpM)+row+(lane%2)*8;
#pragma unroll
        for(int g=0;g<BK/128;++g)
          as_pack[i][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+2*BN)*BK+r*(BK/32)+g*4);
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        int r=BM+wn+j*(32/WarpM)+row;
#pragma unroll
        for(int g=0;g<BK/128;++g) {
          bs_hi_pack[j][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+2*BN)*BK+r*(BK/32)+g*4);
          bs_res_pack[j][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+2*BN)*BK+(r+BN)*(BK/32)+g*4);
        }
      }
#pragma unroll
    for(int kk=0;kk<BK;kk+=32) {
      uint32_t av[TM][4],bv_hi[TN][2],bv_res[TN][2];
      uint8_t as[TM],bs_hi[TN],bs_res[TN];
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
        const auto* ptr_hi=reinterpret_cast<const cute::uint128_t*>(
            sm+BM*BK+sm_index<BK,BN>(wn+j*(32/WarpM)+lane%8,kk+((lane/8)%2)*16));
        const auto* ptr_res=reinterpret_cast<const cute::uint128_t*>(
            sm+(BM+BN)*BK+sm_index<BK,BN>(wn+j*(32/WarpM)+lane%8,kk+((lane/8)%2)*16));
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
  asm volatile("cp.async.wait_group 0;");
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

void gemm_out(at::Tensor const& hi,at::Tensor const& weight,
              at::Tensor const& s_hi,at::Tensor const& s_weight,
              at::Tensor const& res,at::Tensor const& s_res,
              at::Tensor const& out) {
  constexpr int M=32,N=12288,K=2560,BM=128,BN=16,BK=256,Stages=1,WarpM=4;
  TORCH_CHECK(hi.is_cuda() && hi.dim()==2 && hi.size(0)==M && hi.size(1)==K,
              "Expected CUDA hi [32,2560]");
  TORCH_CHECK(res.dim()==2 && res.size(0)==M && res.size(1)==K &&
              weight.dim()==2 && weight.size(0)==N && weight.size(1)==K,
              "Expected res [32,2560], weight [12288,2560]");
  TORCH_CHECK(hi.scalar_type()==at::ScalarType::Float8_e4m3fn &&
              res.scalar_type()==hi.scalar_type() && weight.scalar_type()==hi.scalar_type(),
              "Expected E4M3FN values");
  TORCH_CHECK(s_hi.scalar_type()==at::kByte && s_res.scalar_type()==at::kByte &&
              s_weight.scalar_type()==at::kByte && s_hi.dim()==1 &&
              s_res.dim()==1 && s_weight.dim()==1 && s_hi.numel()==128*(K/32) &&
              s_res.numel()==128*(K/32) && s_weight.numel()==N*(K/32),
              "Expected exact packed E8M0 scale storage");
  TORCH_CHECK(out.scalar_type()==at::kBFloat16 && out.dim()==2 &&
              out.size(0)==M && out.size(1)==N,"Expected BF16 out [32,12288]");
  for(auto const& t:{hi,weight,s_hi,s_weight,res,s_res,out}) {
    TORCH_CHECK(t.device()==hi.device() && t.is_contiguous(),
                "Expected contiguous tensors on one CUDA device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,
                "Expected 16-byte-aligned storage");
  }
  for(auto const& t:{hi,weight,s_hi,s_weight,res,s_res})
    TORCH_CHECK(!out.is_alias_of(t),"Output must not alias inputs");
  c10::cuda::CUDAGuard guard(hi.device());
  cudaDeviceProp props{};
  auto err=cudaGetDeviceProperties(&props,hi.get_device());
  TORCH_CHECK(err==cudaSuccess,"cudaGetDeviceProperties failed: ",cudaGetErrorString(err));
  TORCH_CHECK(props.major==12 && props.minor==0,"Requires SM120a");
  constexpr int Bytes=(((BM+2*BN)*(BK+BK/32)+1023)/1024*1024)*Stages;
  static_assert(Bytes==43008);
  auto fn=gemm<BM,BN,BK,Stages,K,WarpM>;
  auto descriptor=[&](void* ptr,int rows,int tile_rows) {
    CUtensorMap map{};
    uint64_t dimensions[3]={128,uint64_t(rows),uint64_t(K/128)}, strides[2]={uint64_t(K),128};
    uint32_t box[3]={128,uint32_t(tile_rows),BK/128}, element_strides[3]={1,1,1};
    CUresult status=CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,ptr,dimensions,strides,box,element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"TMA descriptor failed: ",int(status));
    return map;
  };
  auto mw=descriptor(weight.data_ptr(),N,BM);
  auto mhi=descriptor(hi.data_ptr(),M,BN);
  auto mres=descriptor(res.data_ptr(),M,BN);
  auto stream=c10::cuda::getCurrentCUDAStream(hi.get_device());
  fn<<<dim3((M+BN-1)/BN,(N+BM-1)/BM),128,Bytes,stream>>>(
    static_cast<const uint8_t*>(s_weight.data_ptr()),
    static_cast<const uint8_t*>(s_hi.data_ptr()),
    static_cast<const uint8_t*>(s_res.data_ptr()),
    static_cast<__nv_bfloat16*>(out.data_ptr()),mw,mhi,mres);
  err=cudaGetLastError();
  TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
}
}

