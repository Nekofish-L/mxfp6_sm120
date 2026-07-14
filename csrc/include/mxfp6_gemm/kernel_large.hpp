#pragma once

#include "mxfp6_gemm/kernel.hpp"

namespace mxfp6_gemm::large_m {

template <class TileM_, class TileN_, class TileK_, class MainloopSchedule_,
          class TileScheduler_ = void>
struct KernelConfig {
  using ElementPairA = mxfp6_gemm::ElementPairA;
  using ElementPairB = mxfp6_gemm::ElementPairB;
  using ElementA = typename ElementPairA::DataType;
  using ElementB = typename ElementPairB::DataType;
  using ElementSF = typename ElementPairA::ScaleFactorType;
  using ElementC = void;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 128;
  static constexpr int AlignmentB = 128;
  static constexpr int AlignmentC = 0;
  static constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using TileShape = cute::Shape<TileM_, TileN_, TileK_>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementPairA,
      LayoutA,
      AlignmentA,
      ElementPairB,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule_>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler_>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using BlockScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
  static constexpr bool IsStreamK =
      cute::is_same_v<TileScheduler_, cutlass::gemm::StreamKScheduler>;
};

using Kernel64x128x128Pingpong = KernelConfig<
    cute::_64, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;
using Kernel128x128x128Pingpong = KernelConfig<
    cute::_128, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;
using Kernel128x128x128Cooperative = KernelConfig<
    cute::_128, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120>;
using Kernel128x128x128StreamK = KernelConfig<
    cute::_128, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::gemm::StreamKScheduler>;
}  // namespace mxfp6_gemm::large_m
