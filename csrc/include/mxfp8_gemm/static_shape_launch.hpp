#pragma once
#include <c10/cuda/CUDACachingAllocator.h>
#include "mxfp8_gemm/validation.hpp"
#include "mxfp8_gemm/pdl.cuh"
namespace mxfp8_static_shape_common {
template<class Kernel,bool Swap,int CtaMultiplier=1>
int64_t launch(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
            at::Tensor const& out,int splits,int swizzle,
            std::optional<at::Tensor> const& persistent, bool query, int sms=0, int raster=0) {
  using Gemm=typename Kernel::Gemm;
  using Config=typename Kernel::BlockScaledConfig;
  int m=Swap?b.size(0):a.size(0), n=Swap?a.size(0):b.size(0), k=a.size(1);
  using PS=typename Kernel::ProblemShape;
  auto problem = [&]() {
    if constexpr(Kernel::IsSwapped && Kernel::MValue>0)
      return PS{m,cute::Int<Kernel::MValue>{},cute::Int<Kernel::KValue>{},cute::_1{}};
    else return PS{m,n,cute::Int<Kernel::KValue>{},cute::_1{}};
  }();
  auto da=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},cute::make_shape(m,k,1));
  auto db=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},cute::make_shape(n,k,1));
  auto dc=cutlass::make_cute_packed_stride(typename Kernel::StrideC{},cute::make_shape(m,n,1));
  auto dd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},cute::make_shape(m,n,1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,problem,
    {reinterpret_cast<typename Kernel::ElementA const*>((Swap?b:a).data_ptr()),da,
     reinterpret_cast<typename Kernel::ElementB const*>((Swap?a:b).data_ptr()),db,
     reinterpret_cast<typename Kernel::ElementSF const*>((Swap?sb:sa).data_ptr()),Config::tile_atom_to_shape_SFA(cute::make_shape(m,n,k,1)),
     reinterpret_cast<typename Kernel::ElementSF const*>((Swap?sa:sb).data_ptr()),Config::tile_atom_to_shape_SFB(cute::make_shape(m,n,k,1))},
    {{1.f,0.f},nullptr,dc,reinterpret_cast<typename Kernel::ElementD*>(out.data_ptr()),dd}};
  args.hw_info.device_id=a.get_device();
  int physical_sms=at::cuda::getDeviceProperties(a.get_device())->multiProcessorCount;
  TORCH_CHECK(sms>=0 && sms<=physical_sms,"Invalid SM count");
  static_assert(CtaMultiplier==1 || !Kernel::IsStreamK,
                "Extra CTA waves are only supported by the static scheduler");
  // Static scheduling uses sm_count as its grid cap. More independent CTAs
  // can fill spare occupancy when a short-K tile uses little shared memory.
  args.hw_info.sm_count=(sms>0?sms:physical_sms)*CtaMultiplier;
  args.scheduler.max_swizzle_size=swizzle;
  using Raster = decltype(args.scheduler.raster_order);
  TORCH_CHECK(raster >= 0 && raster <= 2, "Invalid raster order");
  args.scheduler.raster_order = raster == 1 ? Raster::AlongM :
                               raster == 2 ? Raster::AlongN : Raster::Heuristic;
  if constexpr(Kernel::IsStreamK) {
    using Mode=cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    args.scheduler.splits=splits;
    args.scheduler.decomposition_mode=splits>1?Mode::SplitK:Mode::Heuristic;
  }
  Gemm gemm;
  auto status=gemm.can_implement(args);
  TORCH_CHECK(status==cutlass::Status::kSuccess,"MXFP8 can_implement: ",cutlassGetStatusString(status));
  TORCH_CHECK(Kernel::GemmKernel::SharedStorageSize <=
      at::cuda::getDeviceProperties(a.get_device())->sharedMemPerBlockOptin,
      "MXFP8 can_implement: tile exceeds shared memory capacity");
  int64_t bytes = Gemm::get_workspace_size(args);
  if(query) return bytes;
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
  at::Tensor workspace;
  if(persistent.has_value()) {
    workspace=*persistent;
    TORCH_CHECK(workspace.device()==a.device() && workspace.scalar_type()==at::kByte &&
                workspace.is_contiguous() && workspace.numel()>=bytes,"Invalid MXFP8 workspace");
    using GK=typename Kernel::GemmKernel;
    if constexpr(GK::SharedStorageSize >= (48<<10)) {
      auto error=cudaFuncSetAttribute(cutlass::device_kernel<GK>,cudaFuncAttributeMaxDynamicSharedMemorySize,GK::SharedStorageSize);
      TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
    }
    status=gemm.update(args,workspace.data_ptr());
  } else {
    workspace=at::empty({bytes},a.options().dtype(at::kByte));
    status=gemm.initialize(args,workspace.data_ptr(),stream);
  }
  TORCH_CHECK(status==cutlass::Status::kSuccess,"MXFP8 initialize: ",cutlassGetStatusString(status));
  status=gemm.run(stream, nullptr, mxfp8_runtime::pdl_enabled());
  TORCH_CHECK(status==cutlass::Status::kSuccess,"MXFP8 launch: ",cutlassGetStatusString(status));
  if(bytes>0) c10::cuda::CUDACachingAllocator::recordStream(workspace.storage().data_ptr(),stream);
  return bytes;
}
}
