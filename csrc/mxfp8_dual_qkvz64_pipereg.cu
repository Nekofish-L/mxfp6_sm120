// SPDX-License-Identifier: BSD-3-Clause
// Derived from mxfp6 cfbc5074 csrc/include/mxfp8_gemm/tma256.cuh
// SHA256 4af70650622482c7c8422037b9e71ebfb5b3ad0f533bf3151ee795544b42eb00.
// Isolated M64 QKVZ cfg2; original arithmetic and tile are unchanged.
// Two TMA value stages; packed E8M0 SF is loaded directly into registers.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "cute/arch/mma_sm120.hpp"
#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/arch/barrier.h"

namespace mxfp8_dual_qkvz64_pipereg {
using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,cutlass::float_ue8m0_t,32>;

__device__ __forceinline__ int sf_index(int r,int g,int k) {
  return (r/128)*(k/128)*512+(g/4)*512+(r%32)*16+((r%128)/32)*4+g%4;
}
template<int BK,int Rows>
__device__ __forceinline__ int sm_index(int r,int c) {
  return (c/128)*Rows*128+r*128+((c%128)^((r%8)*16));
}
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
template<int BM,int BN,int BK,int Stages,int FixedK,int WarpM>
__global__ __launch_bounds__(128) void gemm(
    const uint8_t* __restrict__ sw,const uint8_t* __restrict__ shi,const uint8_t* __restrict__ sres,
    __nv_bfloat16* __restrict__ out,
    const __grid_constant__ CUtensorMap mw,const __grid_constant__ CUtensorMap mhi,
    const __grid_constant__ CUtensorMap mres) {
  constexpr int m=12288,n=64,k=FixedK;
  static_assert(m%BM==0 && n%BN==0 && k%BK==0,
                "Global scale packs require full valid M/N/K tiles");
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
          bs_hi_pack[j][g]=*reinterpret_cast<const uint32_t*>(
              shi+sf_index(pn+r,start/32+g*4,k));
          bs_res_pack[j][g]=*reinterpret_cast<const uint32_t*>(
              sres+sf_index(pn+r,start/32+g*4,k));
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

template<int BM,int BN,int BK,int WarpM>
void launch(at::Tensor const& hi,at::Tensor const& weight,
            at::Tensor const& s_hi,at::Tensor const& s_weight,
            at::Tensor const& res,at::Tensor const& s_res,
            at::Tensor const& out) {
  constexpr int M=64,N=12288,K=2560,Stages=2;
  constexpr int Bytes=(BM+2*BN)*BK*Stages;
  static_assert(BM==128 && BN==32 && BK==128 && WarpM==4 && Bytes==49152,
                "Only fixed M64 QKVZ cfg2 with two stages is allowed");
  auto fn=gemm<BM,BN,BK,Stages,K,WarpM>;
  auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes);
  TORCH_CHECK(err==cudaSuccess,"cudaFuncSetAttribute dynamic shared failed: ",cudaGetErrorString(err));
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
  TORCH_CHECK(err==cudaSuccess,"kernel launch failed: ",cudaGetErrorString(err));
}

void gemm_out(at::Tensor const& hi,at::Tensor const& weight,
              at::Tensor const& s_hi,at::Tensor const& s_weight,
              at::Tensor const& res,at::Tensor const& s_res,
              at::Tensor const& out,int64_t config) {
  constexpr int M=64,N=12288,K=2560;
  TORCH_CHECK(hi.is_cuda() && hi.dim()==2 && hi.size(0)==M && hi.size(1)==K,
              "Expected CUDA hi [64,2560]");
  TORCH_CHECK(res.dim()==2 && res.size(0)==M && res.size(1)==K &&
              weight.dim()==2 && weight.size(0)==N && weight.size(1)==K,
              "Expected res [64,2560], weight [12288,2560]");
  TORCH_CHECK(hi.scalar_type()==at::ScalarType::Float8_e4m3fn &&
              res.scalar_type()==hi.scalar_type() && weight.scalar_type()==hi.scalar_type(),
              "Expected E4M3FN values");
  TORCH_CHECK(s_hi.scalar_type()==at::kByte && s_res.scalar_type()==at::kByte &&
              s_weight.scalar_type()==at::kByte && s_hi.dim()==1 &&
              s_res.dim()==1 && s_weight.dim()==1 && s_hi.numel()==128*(K/32) &&
              s_res.numel()==128*(K/32) && s_weight.numel()==N*(K/32),
              "Expected exact packed E8M0 scale storage");
  TORCH_CHECK(out.scalar_type()==at::kBFloat16 && out.dim()==2 &&
              out.size(0)==M && out.size(1)==N,"Expected BF16 out [64,12288]");
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
  TORCH_CHECK(config==2,"Only M64 QKVZ cfg2 is exported");
  launch<128,32,128,4>(hi,weight,s_hi,s_weight,res,s_res,out);
}
std::vector<int64_t> resource_info() {
  constexpr int BM=128,BN=32,BK=128,Stages=2,K=2560,WarpM=4;
  constexpr int DynamicShared=(BM+2*BN)*BK*Stages;
  static_assert(DynamicShared==49152);
  int device=-1;
  auto err=cudaGetDevice(&device);
  TORCH_CHECK(err==cudaSuccess,"cudaGetDevice failed: ",cudaGetErrorString(err));
  int sm_shared=0,max_threads=0,reserved_shared=0,major=0,minor=0;
  err=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);
  TORCH_CHECK(err==cudaSuccess,"cudaDeviceGetAttribute major failed: ",cudaGetErrorString(err));
  err=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);
  TORCH_CHECK(err==cudaSuccess,"cudaDeviceGetAttribute minor failed: ",cudaGetErrorString(err));
  TORCH_CHECK(major==12 && minor==0,"Requires SM120a");
  err=cudaDeviceGetAttribute(&sm_shared,cudaDevAttrMaxSharedMemoryPerMultiprocessor,device);
  TORCH_CHECK(err==cudaSuccess,"cudaDeviceGetAttribute shared/SM failed: ",cudaGetErrorString(err));
  err=cudaDeviceGetAttribute(&max_threads,cudaDevAttrMaxThreadsPerMultiProcessor,device);
  TORCH_CHECK(err==cudaSuccess,"cudaDeviceGetAttribute threads/SM failed: ",cudaGetErrorString(err));
  err=cudaDeviceGetAttribute(&reserved_shared,cudaDevAttrReservedSharedMemoryPerBlock,device);
  TORCH_CHECK(err==cudaSuccess,"cudaDeviceGetAttribute reserved shared/block failed: ",cudaGetErrorString(err));
  auto fn=gemm<BM,BN,BK,Stages,K,WarpM>;
  err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,DynamicShared);
  TORCH_CHECK(err==cudaSuccess,"cudaFuncSetAttribute dynamic shared failed: ",cudaGetErrorString(err));
  cudaFuncAttributes attr{};
  err=cudaFuncGetAttributes(&attr,fn);
  TORCH_CHECK(err==cudaSuccess,"cudaFuncGetAttributes failed: ",cudaGetErrorString(err));
  int active=0;
  err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,128,DynamicShared);
  TORCH_CHECK(err==cudaSuccess,"cudaOccupancyMaxActiveBlocksPerMultiprocessor failed: ",
              cudaGetErrorString(err));
  return {int64_t(attr.numRegs),int64_t(attr.sharedSizeBytes),int64_t(DynamicShared),
          int64_t(active),int64_t(sm_shared),int64_t(max_threads),
          int64_t(reserved_shared)};
}
}

