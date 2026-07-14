#pragma once

#include "mxfp6_gemm/kernel.hpp"

namespace mxfp6_gemm::swap_m32 {

// Compute D.T = B @ A.T. This turns the small batch dimension into the MMA N
// dimension and stores column-major [N,M] directly into row-major [M,N].
template <class TileM_, class TileN_, class TileK_, class MainloopSchedule_,
          class EpilogueTile_ = cutlass::epilogue::collective::EpilogueTileAuto,
          class TileScheduler_ = void, class StageCount_ = void>
struct KernelConfig {
  using ElementPairA = mxfp6_gemm::ElementPairB;
  using ElementPairB = mxfp6_gemm::ElementPairA;
  using ElementA = typename ElementPairA::DataType;
  using ElementB = typename ElementPairB::DataType;
  using ElementSF = typename ElementPairA::ScaleFactorType;
  using ElementC = void;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  static constexpr int AlignmentA = 128;
  static constexpr int AlignmentB = 128;
  static constexpr int AlignmentC = 0;
  static constexpr int AlignmentD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using TileShape = cute::Shape<TileM_, TileN_, TileK_>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using MainloopSchedule = MainloopSchedule_;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      EpilogueTile_,
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
      cute::conditional_t<
          cute::is_void_v<StageCount_>,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          StageCount_>,
      MainloopSchedule>::CollectiveOp;

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

using KernelM128N8Cooperative = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120>;
using KernelM128N16Cooperative = KernelConfig<
    cute::_128, cute::_16, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120>;
using KernelM128N8StreamK = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StreamKScheduler>;
using KernelM128N16StreamK = KernelConfig<
    cute::_128, cute::_16, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StreamKScheduler>;
using KernelM64N16Pingpong = KernelConfig<
    cute::_64, cute::_16, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;
using KernelM64N16K256Pingpong = KernelConfig<
    cute::_64, cute::_16, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;
using KernelM64N16K512Pingpong = KernelConfig<
    cute::_64, cute::_16, cute::_512,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;

}  // namespace mxfp6_gemm::swap_m32
