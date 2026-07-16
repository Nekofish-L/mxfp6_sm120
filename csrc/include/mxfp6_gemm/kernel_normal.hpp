#pragma once

#include "mxfp6_gemm/kernel.hpp"

namespace mxfp6_gemm::normal {

template <class TileM_, class TileN_, class TileK_, class MainloopSchedule_,
          class TileScheduler_ = void, class StageCount_ = void,
          class ElementPairA_ = mxfp6_gemm::ElementPairA,
          class ElementPairB_ = mxfp6_gemm::ElementPairB>
struct KernelConfig {
  using ElementPairA = ElementPairA_;
  using ElementPairB = ElementPairB_;
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
      cute::conditional_t<
          cute::is_void_v<StageCount_>,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          StageCount_>,
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

using Kernel64x16x256Pingpong = KernelConfig<
    cute::_64, cute::_16, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;
using Kernel64x32x128Stage3Pingpong = KernelConfig<
    cute::_64, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    void, cutlass::gemm::collective::StageCount<3>>;
using Kernel64x64x128Stage2Pingpong = KernelConfig<
    cute::_64, cute::_64, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    void, cutlass::gemm::collective::StageCount<2>>;
using Kernel64x64x128Stage4Pingpong = KernelConfig<
    cute::_64, cute::_64, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    void, cutlass::gemm::collective::StageCount<4>>;
using Kernel128x128x128Pingpong = KernelConfig<
    cute::_128, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;
using Kernel128x16x128Stage3StreamK = KernelConfig<
    cute::_128, cute::_16, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::gemm::StreamKScheduler,
    cutlass::gemm::collective::StageCount<3>>;
using Kernel128x32x128StreamK = KernelConfig<
    cute::_128, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::gemm::StreamKScheduler>;
using Kernel128x64x128StreamK = KernelConfig<
    cute::_128, cute::_64, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::gemm::StreamKScheduler>;
using Kernel128x128x128StreamK = KernelConfig<
    cute::_128, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::gemm::StreamKScheduler>;

// Exact target-shape winners retained alongside the general portfolio.
using TargetKernel64x64x128Pingpong = KernelConfig<
    cute::_64, cute::_64, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120>;

// Humming-style large-M path: preserve the public packed-W6 activation and
// weight representation, expand only the activation into a temporary E4M3
// tensor, and issue SM120's faster E4M3 x E3M2 mixed MMA. The weight remains
// physically packed at six bits throughout.
using ElementPairA8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using KernelW6A8_64x64x128Pingpong = KernelConfig<
    cute::_64, cute::_64, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    void, void, ElementPairA8>;
using KernelW6A8_64x64x128StaticPingpong = KernelConfig<
    cute::_64, cute::_64, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::gemm::StaticPersistentScheduler, void, ElementPairA8>;
using KernelW6A8_64x64x128Stage4StaticPingpong = KernelConfig<
    cute::_64, cute::_64, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<4>, ElementPairA8>;
using KernelW6A8_64x64x256StaticPingpong = KernelConfig<
    cute::_64, cute::_64, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::gemm::StaticPersistentScheduler, void, ElementPairA8>;
using KernelW6A8_64x128x128Pingpong = KernelConfig<
    cute::_64, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    void, void, ElementPairA8>;
using KernelW6A8_64x128x128StaticPingpong = KernelConfig<
    cute::_64, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::gemm::StaticPersistentScheduler, void, ElementPairA8>;
using KernelW6A8_128x128x128Pingpong = KernelConfig<
    cute::_128, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    void, void, ElementPairA8>;
using KernelW6A8_128x128x128Cooperative = KernelConfig<
    cute::_128, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    void, void, ElementPairA8>;
using KernelW6A8_128x128x128StreamK = KernelConfig<
    cute::_128, cute::_128, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::gemm::StreamKScheduler, void, ElementPairA8>;
}  // namespace mxfp6_gemm::normal
