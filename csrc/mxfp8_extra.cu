#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "cutlass/util/packed_stride.hpp"
#include "mxfp6_gemm/kernel_swapped.hpp"
#include "mxfp8_gemm/launch.hpp"

namespace mxfp8_extra {
using FP8=cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Coop=cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120;
using Ping=cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120;
using Static=cutlass::gemm::StaticPersistentScheduler;
using StreamK=cutlass::gemm::StreamKScheduler;
template<int M,int N,int K,class Schedule=Coop,class Scheduler=Static,class Stages=void>
using Kernel=mxfp6_gemm::swapped::KernelConfig<cute::Int<M>,cute::Int<N>,cute::Int<K>,Schedule,
    cutlass::epilogue::collective::EpilogueTileAuto,Scheduler,Stages,FP8,FP8,cutlass::bfloat16_t>;
int64_t mm_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
              at::Tensor const& out,int64_t tactic,int64_t splits,int64_t swizzle,
              std::optional<at::Tensor> persistent,bool query,int64_t sms) {
  // Tactic aliases 300+base and 400+base select explicit raster directions.
  int raster = tactic >= 300 && tactic < 500 ? int(tactic / 100) - 2 : 0;
  if (raster) tactic %= 100;
  mxfp8_common::validate(a,b,sa,sb,out);
  mxfp8_common::validate_scheduler(splits,swizzle);
  c10::cuda::CUDAGuard guard(a.device());
#define RUN(ID,...) case ID:return mxfp8_common::launch<__VA_ARGS__,true>(a,b,sa,sb,out,splits,swizzle,persistent,query,sms,raster)
  switch(tactic) {
    RUN(51,Kernel<128,64,128,Coop,StreamK>);
    RUN(52,Kernel<64,64,128,Ping>);
    RUN(53,Kernel<64,64,128,Ping,Static,cutlass::gemm::collective::StageCount<3>>);
    RUN(54,Kernel<64,64,256,Ping>);
    RUN(55,Kernel<64,128,128,Ping>);
    RUN(56,Kernel<128,128,128>);
    RUN(57,Kernel<128,128,128,Coop,StreamK>);
    default:TORCH_CHECK(false,"Unknown transposed tactic");
  }
#undef RUN
}
}
