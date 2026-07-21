#pragma once

#include "mxfp6_gemm/kernel.hpp"

namespace mxfp6_gemm::swapped {

// Compute D.T = B @ A.T. This turns the small batch dimension into the MMA N
// dimension and stores column-major [N,M] directly into row-major [M,N].
template <class TileM_, class TileN_, class TileK_, class MainloopSchedule_,
          class EpilogueTile_ = cutlass::epilogue::collective::EpilogueTileAuto,
          class TileScheduler_ = void, class StageCount_ = void,
          class ElementPairA_ = mxfp6_gemm::ElementPairB,
          class ElementPairB_ = mxfp6_gemm::ElementPairA>
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

using Kernel128x8Stage2Cooperative = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, cutlass::gemm::collective::StageCount<2>>;
using Kernel128x8Stage4Cooperative = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, cutlass::gemm::collective::StageCount<4>>;
using Kernel128x8StreamK = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StreamKScheduler>;
using Kernel128x16StreamK = KernelConfig<
    cute::_128, cute::_16, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StreamKScheduler>;
using Kernel128x32Stage3StreamK = KernelConfig<
    cute::_128, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StreamKScheduler,
    cutlass::gemm::collective::StageCount<3>>;
using Kernel128x64StreamK = KernelConfig<
    cute::_128, cute::_64, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StreamKScheduler>;
using Kernel64x16x128Stage3Pingpong = KernelConfig<
    cute::_64, cute::_16, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, cutlass::gemm::collective::StageCount<3>>;
using Kernel64x16x256Stage3Pingpong = KernelConfig<
    cute::_64, cute::_16, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, cutlass::gemm::collective::StageCount<3>>;
using Kernel64x32x128Stage3Pingpong = KernelConfig<
    cute::_64, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, cutlass::gemm::collective::StageCount<3>>;

// Exact target-shape winners retained alongside the general portfolio.
using TargetKernel128x8Stage4StaticCooperative = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<4>>;
using TargetKernel128x8StaticCooperative = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler>;
using TargetKernel128x32Cooperative = KernelConfig<
    cute::_128, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120>;
using TargetKernel128x32StaticCooperative = KernelConfig<
    cute::_128, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler>;
using TargetKernel64x16x128Stage6StaticPingpong = KernelConfig<
    cute::_64, cute::_16, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<6>>;
using TargetKernel64x16x256Pingpong = KernelConfig<
    cute::_64, cute::_16, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler>;
using TargetKernel64x16x256Stage3StaticPingpong = KernelConfig<
    cute::_64, cute::_16, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<3>>;
using TargetKernel64x16x512Pingpong = KernelConfig<
    cute::_64, cute::_16, cute::_512,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler>;

// Mixed path for D.T = W6 @ A8.T. Unlike the normal W6A8 orientation this
// keeps the small logical batch in the MMA N dimension, so M=1/16 does not
// waste a 64- or 128-row activation tile. Persistent weights remain packed
// E3M2; only the transient activation uses byte-addressable E4M3 storage.
using ElementPairB8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using KernelW6A8_128x8StaticCooperative = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler, void,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_128x8Stage4StaticCooperative = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<4>,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
// Profiler-selected small-batch portfolio. M=1 uses a narrow cooperative
// tile (or Stream-K for deep K), while M=16 uses a full 16-column tile and
// deeper K tiles to remove the former W6A8 decode penalty.
using KernelW6A8_128x8Stage5StaticCooperative = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<5>,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_128x8StreamK = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StreamKScheduler, void,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_128x16Stage4StaticCooperative = KernelConfig<
    cute::_128, cute::_16, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<4>,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_64x16x256StaticPingpong = KernelConfig<
    cute::_64, cute::_16, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler, void,
    mxfp6_gemm::ElementPairB, ElementPairB8>;

// M=32 portfolio. In the swapped orientation the logical activation batch is
// the tile N dimension, so these kernels cover all 32 rows without either the
// two-wave x16 launch or the half-empty normal 64-row tile used previously.
using KernelW6A8_64x32x128Pingpong = KernelConfig<
    cute::_64, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, void, mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_64x32x128Stage3Pingpong = KernelConfig<
    cute::_64, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, cutlass::gemm::collective::StageCount<3>,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_64x32x128StaticPingpong = KernelConfig<
    cute::_64, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler, void,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_64x32x128Stage3StaticPingpong = KernelConfig<
    cute::_64, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<3>,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_64x32x256Pingpong = KernelConfig<
    cute::_64, cute::_32, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, void, mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_64x32x256Stage2Pingpong = KernelConfig<
    cute::_64, cute::_32, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, cutlass::gemm::collective::StageCount<2>,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_64x32x256StaticPingpong = KernelConfig<
    cute::_64, cute::_32, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler, void,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_64x32x256Stage2StaticPingpong = KernelConfig<
    cute::_64, cute::_32, cute::_256,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<2>,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_128x32Cooperative = KernelConfig<
    cute::_128, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, void, mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_128x32Stage2Cooperative = KernelConfig<
    cute::_128, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    void, cutlass::gemm::collective::StageCount<2>,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_128x32StaticCooperative = KernelConfig<
    cute::_128, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler, void,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
using KernelW6A8_128x32Stage2StaticCooperative = KernelConfig<
    cute::_128, cute::_32, cute::_128,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::gemm::StaticPersistentScheduler,
    cutlass::gemm::collective::StageCount<2>,
    mxfp6_gemm::ElementPairB, ElementPairB8>;
}  // namespace mxfp6_gemm::swapped
