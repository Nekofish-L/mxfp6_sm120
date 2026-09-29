#ifndef NDEBUG
#define NDEBUG
#endif
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "cutlass/util/packed_stride.hpp"
#include "mxfp6_gemm/kernel.hpp"
#include "mxfp8_gemm/launch.hpp"

namespace mxfp8_release_direct {
using FP8=cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Coop=cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120;
using Ping=cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120;
template<int M,int N,int K,bool Swap,class Schedule=Coop,class Stages=void>
struct KernelConfig {
  using ElementA=cutlass::float_e4m3_t;
  using ElementB=ElementA;
  using ElementSF=cutlass::float_ue8m0_t;
  using ElementC=void;
  using ElementD=cutlass::bfloat16_t;
  using Tile=cute::Shape<cute::Int<M>,cute::Int<N>,cute::Int<K>>;
  using Cluster=cute::Shape<cute::_1,cute::_1,cute::_1>;
  using LayoutD=cute::conditional_t<Swap,cutlass::layout::ColumnMajor,cutlass::layout::RowMajor>;
  // The generic register-to-global epilogue does not require Hopper MMA.
  using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tile,Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,float,float,void,LayoutD,0,
    ElementD,LayoutD,8,cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp;
  using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,cutlass::arch::OpClassBlockScaledTensorOp,
    FP8,cutlass::layout::RowMajor,128,FP8,cutlass::layout::ColumnMajor,128,
    float,Tile,Cluster,cute::conditional_t<cute::is_void_v<Stages>,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,Stages>,Schedule>::CollectiveOp;
  using BaseKernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int,int>,Mainloop,Epi,cutlass::gemm::StaticPersistentScheduler>;
  struct GemmKernel : BaseKernel {};
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using BlockScaledConfig=typename Mainloop::Sm1xxBlkScaledConfig;
  using StrideA=typename GemmKernel::StrideA;
  using StrideB=typename GemmKernel::StrideB;
  using StrideC=typename GemmKernel::StrideC;
  using StrideD=typename GemmKernel::StrideD;
  static constexpr bool IsStreamK=false;
};
using mxfp8_common::launch;

int64_t mm_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
              at::Tensor const& out,int64_t tactic,int64_t splits,int64_t swizzle,
              std::optional<at::Tensor> persistent,bool query,int64_t sms) {
  TORCH_CHECK(tactic>=1300 && tactic<1600,"Unknown release direct tactic");
  int direction=int(tactic/100)-13;
  tactic=tactic%100+(direction?200+direction*100:0);
  // Tactic aliases 300+base and 400+base select explicit raster directions.
  int raster = tactic >= 300 && tactic < 500 ? int(tactic / 100) - 2 : 0;
  if (raster) tactic %= 100;
  mxfp8_common::validate(a,b,sa,sb,out);
  mxfp8_common::validate_scheduler(splits,swizzle);
  c10::cuda::CUDAGuard guard(a.device());
#define RUN(ID,SWAP,...) case ID:return launch<__VA_ARGS__,SWAP>(a,b,sa,sb,out,splits,swizzle,persistent,query,sms,raster)
#define RUN2(ID,SWAP,...) case ID:return launch<__VA_ARGS__,SWAP,2>(a,b,sa,sb,out,splits,swizzle,persistent,query,sms,raster)
  using S2=cutlass::gemm::collective::StageCount<2>;
  using S3=cutlass::gemm::collective::StageCount<3>;
  switch(tactic) {
    RUN(43,true,KernelConfig<64,16,256,true,Ping>);
    RUN(44,true,KernelConfig<128,8,128,true>);
    RUN(45,true,KernelConfig<64,32,128,true,Ping>);
    RUN(46,true,KernelConfig<128,64,128,true>);
    RUN(47,false,KernelConfig<128,32,128,false>);
    RUN(48,false,KernelConfig<64,64,128,false,Ping>);
    RUN(49,false,KernelConfig<128,128,128,false>);
    RUN(50,false,KernelConfig<256,64,128,false>);
    RUN2(58,true,KernelConfig<128,8,128,true,Coop,S2>);
    RUN2(59,true,KernelConfig<128,16,128,true,Coop,S2>);
    RUN2(60,true,KernelConfig<128,32,128,true,Coop,S2>);
    RUN2(61,true,KernelConfig<64,32,128,true,Ping,S2>);
    RUN2(62,false,KernelConfig<64,64,128,false,Ping,S2>);
    RUN2(63,false,KernelConfig<64,128,128,false,Ping,S2>);
    RUN(64,true,KernelConfig<128,16,128,true,Coop,S2>);
    RUN(65,true,KernelConfig<128,32,128,true,Coop,S2>);
    RUN2(66,true,KernelConfig<32,16,256,true,Ping,S2>);
    RUN2(67,true,KernelConfig<32,16,256,true,Ping,S3>);
    RUN2(68,true,KernelConfig<32,32,256,true,Ping,S2>);
    RUN2(69,true,KernelConfig<32,32,256,true,Ping,S3>);
    // Match the small-batch Triton tile/epilogue choices using CUTLASS TMA.
    // Two waves permit multiple resident CTAs when shared memory allows it.
    RUN2(86,true,KernelConfig<64,64,256,true,Ping,S2>);
    RUN2(87,false,KernelConfig<64,64,256,false,Ping,S2>);
    RUN2(88,true,KernelConfig<64,64,128,true,Ping,S3>);
    RUN2(89,false,KernelConfig<64,64,128,false,Ping,S3>);
    RUN2(90,true,KernelConfig<128,16,128,true,Coop,S3>);
    RUN2(91,false,KernelConfig<64,128,128,false,Ping,S3>);
    RUN2(92,true,KernelConfig<64,32,256,true,Ping,S2>);
    RUN2(93,true,KernelConfig<64,16,256,true,Ping,S2>);
    RUN2(94,true,KernelConfig<128,32,128,true,Coop,S3>);
    RUN2(95,true,KernelConfig<64,64,128,true,Ping,S2>);
    default:TORCH_CHECK(false,"Unknown direct-store tactic");
  }
#undef RUN
#undef RUN2
}
}
