#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "cutlass/util/packed_stride.hpp"
#include "mxfp6_gemm/kernel_swapped.hpp"
#include "mxfp6_gemm/kernel_normal.hpp"
#include "mxfp8_gemm/launch.hpp"

namespace mxfp8_release {
using FP8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Coop = cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120;
using Dynamic = cutlass::gemm::PersistentScheduler;

// A distinct device type prevents symbol sharing with the debug build.
template<class Base> struct ReleaseDevice : Base {};
template<class Base> struct ReleaseConfig : Base {
  using GemmKernel = ReleaseDevice<typename Base::GemmKernel>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template<int M, int N, class Schedule=Coop, class Stages=void, class Scheduler=Dynamic>
using Swapped = mxfp6_gemm::swapped::KernelConfig<cute::Int<M>,cute::Int<N>,cute::_128,
    Schedule,cutlass::epilogue::collective::EpilogueTileAuto,Scheduler,Stages,
    FP8,FP8,cutlass::bfloat16_t>;
template<int M, int N, class Schedule=Coop, class Stages=void, class Scheduler=Dynamic>
using Normal = mxfp6_gemm::normal::KernelConfig<cute::Int<M>,cute::Int<N>,cute::_128,
    Schedule,Scheduler,Stages,FP8,FP8,cutlass::bfloat16_t>;

int64_t mm_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
               at::Tensor const& out,int64_t config) {
  mxfp8_common::validate(a,b,sa,sb,out);
  c10::cuda::CUDAGuard guard(a.device());
#define RUN(ID,SWAP,...) case ID: return mxfp8_common::launch<ReleaseConfig<__VA_ARGS__>,SWAP>(a,b,sa,sb,out,1,1,std::nullopt,false,0)
  switch(config) {
    RUN(0,true,Swapped<128,8,Ping,void>);
    RUN(1,true,Swapped<128,16,Ping,void>);
    RUN(2,true,Swapped<128,32,Ping,void>);
    RUN(3,true,Swapped<128,64,Ping,void>);
    RUN(4,true,Swapped<128,8,Ping,cutlass::gemm::collective::StageCount<2>>);
    RUN(5,true,Swapped<128,16,Ping,cutlass::gemm::collective::StageCount<2>>);
    RUN(6,true,Swapped<128,32,Ping,cutlass::gemm::collective::StageCount<2>>);
    RUN(7,true,Swapped<128,64,Ping,cutlass::gemm::collective::StageCount<2>>);
    RUN(8,true,Swapped<128,8,Coop,cutlass::gemm::collective::StageCount<2>>);
    RUN(9,true,Swapped<128,16,Coop,cutlass::gemm::collective::StageCount<2>>);
    RUN(10,true,Swapped<128,32,Coop,cutlass::gemm::collective::StageCount<2>>);
    RUN(11,true,Swapped<128,64,Coop,cutlass::gemm::collective::StageCount<2>>);
    RUN(12,true,Swapped<128,8,Coop,cutlass::gemm::collective::StageCount<3>>);
    RUN(13,true,Swapped<128,16,Coop,cutlass::gemm::collective::StageCount<3>>);
    RUN(14,true,Swapped<128,32,Coop,cutlass::gemm::collective::StageCount<3>>);
    RUN(15,true,Swapped<128,64,Coop,cutlass::gemm::collective::StageCount<3>>);
    default:TORCH_CHECK(false,"Unknown schedule configuration");
  }
#undef RUN
}
}
