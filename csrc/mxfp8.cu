#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "cutlass/util/packed_stride.hpp"
#include "mxfp6_gemm/kernel_swapped.hpp"
#include "mxfp6_gemm/kernel_normal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "mxfp8_gemm/launch.hpp"

namespace mxfp8_sm120 {
using FP8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Coop = cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120;
using Static = cutlass::gemm::StaticPersistentScheduler;
using StreamK = cutlass::gemm::StreamKScheduler;
template<int M,int N,int K,class Schedule=Coop,class Scheduler=Static,class Stages=void>
using Swapped = mxfp6_gemm::swapped::KernelConfig<cute::Int<M>,cute::Int<N>,cute::Int<K>,Schedule,
    cutlass::epilogue::collective::EpilogueTileAuto,Scheduler,Stages,FP8,FP8,cutlass::bfloat16_t>;
template<int M,int N,int K,class Schedule=Ping,class Scheduler=Static,class Stages=void>
using Normal = mxfp6_gemm::normal::KernelConfig<cute::Int<M>,cute::Int<N>,cute::Int<K>,Schedule,
    Scheduler,Stages,FP8,FP8,cutlass::bfloat16_t>;

using mxfp8_common::launch;

int64_t mm_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
            at::Tensor const& out,int64_t tactic,int64_t splits,int64_t swizzle,
            std::optional<at::Tensor> persistent, bool query, int64_t sms) {
  mxfp8_common::validate(a,b,sa,sb,out);
  mxfp8_common::validate_scheduler(splits,swizzle);
  c10::cuda::CUDAGuard guard(a.device());
#define RUN(ID,SWAP,...) case ID: return launch<__VA_ARGS__,SWAP>(a,b,sa,sb,out,splits,swizzle,persistent,query,sms)
  switch(tactic) {
    RUN(0,true,Swapped<128,8,128>);
    RUN(1,true,Swapped<128,16,128>);
    RUN(2,true,Swapped<64,16,256,Ping>);
    RUN(3,true,Swapped<64,32,128,Ping>);
    RUN(4,true,Swapped<64,32,256,Ping>);
    RUN(5,true,Swapped<128,32,128>);
    RUN(6,true,Swapped<128,64,128>);
    RUN(7,false,Normal<64,64,128>);
    RUN(8,false,Normal<64,128,128>);
    RUN(9,false,Normal<128,128,128>);
    RUN(10,false,Normal<128,128,128,Coop>);
    RUN(11,false,Normal<128,64,128,Coop>);
    RUN(12,true,Swapped<128,8,128,Coop,StreamK>);
    RUN(13,true,Swapped<128,16,128,Coop,StreamK>);
    RUN(14,true,Swapped<128,32,128,Coop,StreamK>);
    RUN(15,false,Normal<128,64,128,Coop,StreamK>);
    RUN(16,false,Normal<128,128,128,Coop,StreamK>);
    RUN(17,true,Swapped<64,16,128,Ping,Static,cutlass::gemm::collective::StageCount<3>>);
    RUN(18,true,Swapped<64,16,256,Ping,Static,cutlass::gemm::collective::StageCount<2>>);
    RUN(19,true,Swapped<64,32,128,Ping,Static,cutlass::gemm::collective::StageCount<3>>);
    RUN(20,true,Swapped<64,32,256,Ping,Static,cutlass::gemm::collective::StageCount<2>>);
    RUN(21,true,Swapped<128,8,128,Coop,Static,cutlass::gemm::collective::StageCount<4>>);
    RUN(22,true,Swapped<128,16,128,Coop,Static,cutlass::gemm::collective::StageCount<4>>);
    RUN(23,false,Normal<64,64,128,Ping,Static,cutlass::gemm::collective::StageCount<2>>);
    RUN(24,false,Normal<64,64,128,Ping,Static,cutlass::gemm::collective::StageCount<3>>);
    RUN(25,false,Normal<64,128,128,Ping,Static,cutlass::gemm::collective::StageCount<2>>);
    RUN(26,false,Normal<64,128,128,Ping,Static,cutlass::gemm::collective::StageCount<3>>);
    RUN(27,false,Normal<128,128,128,Ping,Static,cutlass::gemm::collective::StageCount<2>>);
    RUN(28,false,Normal<128,128,128,Coop,Static,cutlass::gemm::collective::StageCount<2>>);
    RUN(29,false,Normal<128,64,128,Coop,Static,cutlass::gemm::collective::StageCount<2>>);
    RUN(31,false,Normal<64,64,256,Ping>);
    RUN(32,true,Swapped<128,64,128,Coop,Static,cutlass::gemm::collective::StageCount<2>>);
    default: TORCH_CHECK(false,"Unknown MXFP8 tactic ",tactic);
  }
#undef RUN
}
} // namespace mxfp8_sm120
TORCH_LIBRARY(mxfp8_sm120,m) {
  m.def("mm_out(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor(a!) out, int tactic, int splits=1, int swizzle=1, Tensor? workspace=None, bool query=False, int sms=0) -> int");
}
TORCH_LIBRARY_IMPL(mxfp8_sm120,CUDA,m) { m.impl("mm_out",mxfp8_sm120::mm_out); }
