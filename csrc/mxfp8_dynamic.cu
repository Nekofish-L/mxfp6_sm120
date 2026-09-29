#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "cutlass/util/packed_stride.hpp"
#include "mxfp6_gemm/kernel_swapped.hpp"
#include "mxfp6_gemm/kernel_normal.hpp"
#include "mxfp8_gemm/launch.hpp"

namespace mxfp8_dynamic {
using FP8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Coop = cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120;
using Dynamic = cutlass::gemm::PersistentScheduler;

template<int M, int N, class Schedule=Coop, class Stages=void, class Scheduler=Dynamic>
using Swapped = mxfp6_gemm::swapped::KernelConfig<cute::Int<M>,cute::Int<N>,cute::_128,
    Schedule,cutlass::epilogue::collective::EpilogueTileAuto,Scheduler,Stages,
    FP8,FP8,cutlass::bfloat16_t>;
template<int M, int N, class Schedule=Coop, class Stages=void, class Scheduler=Dynamic>
using Normal = mxfp6_gemm::normal::KernelConfig<cute::Int<M>,cute::Int<N>,cute::_128,
    Schedule,Scheduler,Stages,FP8,FP8,cutlass::bfloat16_t>;

int64_t mm_out(at::Tensor const& a, at::Tensor const& b,
               at::Tensor const& sa, at::Tensor const& sb, at::Tensor const& out,
               int64_t tactic, int64_t splits, int64_t swizzle,
               std::optional<at::Tensor> workspace, bool query, int64_t sms) {
  // Tactic aliases 300+base and 400+base select explicit raster directions.
  int raster = tactic >= 300 && tactic < 500 ? int(tactic / 100) - 2 : 0;
  if (raster) tactic %= 100;
  mxfp8_common::validate(a,b,sa,sb,out);
  mxfp8_common::validate_scheduler(splits,swizzle);
  c10::cuda::CUDAGuard guard(a.device());
#define RUN(ID,SWAP,...) case ID: return mxfp8_common::launch<__VA_ARGS__,SWAP>(a,b,sa,sb,out,splits,swizzle,workspace,query,sms,raster)
  switch (tactic) {
    RUN(70,true,Swapped<128,8>);
    RUN(71,true,Swapped<128,16>);
    RUN(72,true,Swapped<128,32>);
    RUN(73,true,Swapped<128,64>);
    RUN(74,true,Swapped<64,16,Ping>);
    RUN(75,true,Swapped<64,32,Ping>);
    RUN(76,false,Normal<64,64,Ping>);
    RUN(77,false,Normal<64,128,Ping>);
    RUN(78,false,Normal<128,128>);
    RUN(79,false,Normal<128,128,Ping>);
    default: TORCH_CHECK(false,"Unknown dynamic MXFP8 tactic ",tactic);
  }
#undef RUN
}
}
