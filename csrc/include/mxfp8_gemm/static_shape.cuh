#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "cutlass/util/packed_stride.hpp"
#include "mxfp6_gemm/kernel_swapped.hpp"
#include "mxfp6_gemm/kernel_normal.hpp"
#include "mxfp8_gemm/static_shape_launch.hpp"
namespace mxfp8_static_shape {
using FP8=cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Coop=cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120;
using Dynamic=cutlass::gemm::PersistentScheduler;
template<int TM,int TN,bool Swap,int K,int M=0>
struct Kernel : cute::conditional_t<Swap,
 mxfp6_gemm::swapped::KernelConfig<cute::Int<TM>,cute::Int<TN>,cute::_128,Coop,cutlass::epilogue::collective::EpilogueTileAuto,Dynamic,void,FP8,FP8,cutlass::bfloat16_t>,
 mxfp6_gemm::normal::KernelConfig<cute::Int<TM>,cute::Int<TN>,cute::_128,Coop,Dynamic,void,FP8,FP8,cutlass::bfloat16_t>> {
 using Base=cute::conditional_t<Swap,
 mxfp6_gemm::swapped::KernelConfig<cute::Int<TM>,cute::Int<TN>,cute::_128,Coop,cutlass::epilogue::collective::EpilogueTileAuto,Dynamic,void,FP8,FP8,cutlass::bfloat16_t>,
 mxfp6_gemm::normal::KernelConfig<cute::Int<TM>,cute::Int<TN>,cute::_128,Coop,Dynamic,void,FP8,FP8,cutlass::bfloat16_t>>;
 static constexpr int KValue=K,MValue=M;
 static constexpr bool IsSwapped=Swap;
 using ProblemShape=cute::Shape<int,cute::conditional_t<Swap && (M>0),cute::Int<M>,int>,cute::Int<K>,cute::_1>;
 using GemmKernel=cutlass::gemm::kernel::GemmUniversal<ProblemShape,typename Base::CollectiveMainloop,typename Base::CollectiveEpilogue,Dynamic>;
 using Gemm=cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};
template<int TN,int K,int M=0>
int64_t run(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,at::Tensor const& out) {
 return mxfp8_static_shape_common::launch<Kernel<128,TN,true,K,M>,true>(a,b,sa,sb,out,1,1,std::nullopt,false,0);
}
template<int TN,int K>
int64_t exact(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,at::Tensor const& out) {
#define M_CASE(M) case M:return run<TN,K,M>(a,b,sa,sb,out)
 switch(a.size(0)) {M_CASE(10);M_CASE(12);M_CASE(14);M_CASE(16);M_CASE(24);M_CASE(32);M_CASE(40);default:return run<TN,K>(a,b,sa,sb,out);}
#undef M_CASE
}
template<int K>
int64_t dispatch(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,at::Tensor const& out,int config) {
 switch(config) {
 case 0:return run<32,K>(a,b,sa,sb,out);
 case 1:return exact<32,K>(a,b,sa,sb,out);
 case 2:return run<16,K>(a,b,sa,sb,out);
 case 3:return exact<16,K>(a,b,sa,sb,out);
 case 4:return run<64,K>(a,b,sa,sb,out);
 case 5:return mxfp8_static_shape_common::launch<Kernel<128,128,false,K>,false>(a,b,sa,sb,out,1,1,std::nullopt,false,0);
 default:TORCH_CHECK(false,"Unknown static-shape config");
 }
}
int64_t mm_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,at::Tensor const& out,int64_t config) {
 mxfp8_common::validate(a,b,sa,sb,out);c10::cuda::CUDAGuard guard(a.device());
 if(a.size(1)==2048)return dispatch<2048>(a,b,sa,sb,out,config);
 if(a.size(1)==2560)return dispatch<2560>(a,b,sa,sb,out,config);
 TORCH_CHECK(false,"Static-shape experiment requires K=2048 or K=2560");
}
}
