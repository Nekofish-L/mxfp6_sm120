#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cute/tensor.hpp"

namespace mxfp6_gemm::grouped {

using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

template <class TileM_, class TileN_, class TileK_,
          class MainloopSchedule_ =
              cutlass::gemm::collective::KernelScheduleAuto,
          class StageCount_ = void, class ElementD_ = cutlass::half_t,
          class LayoutA_ = cutlass::layout::RowMajor,
          class EpilogueTile_ =
              cutlass::epilogue::collective::EpilogueTileAuto,
          class EpilogueSchedule_ =
              cutlass::epilogue::collective::EpilogueScheduleAuto>
struct KernelConfig {
  // The grouped kernel evaluates D.T = W6 @ A8.T. Keeping the routed-token
  // dimension in MMA N avoids wasting a 64/128-row tile during decode.
  using ElementPairA = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
  using ElementPairB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementA = typename ElementPairA::DataType;
  using ElementB = typename ElementPairB::DataType;
  using ElementSF = typename ElementPairA::ScaleFactorType;
  using ElementC = void;
  using ElementD = ElementD_;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = LayoutA_;
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

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          EpilogueTile_,
          ElementAccumulator, ElementCompute, ElementC, LayoutC*, AlignmentC,
          ElementD, LayoutD*, AlignmentD,
          EpilogueSchedule_>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementPairA, LayoutA*, AlignmentA,
          ElementPairB, LayoutB*, AlignmentB, ElementAccumulator, TileShape,
          ClusterShape,
          cute::conditional_t<
              cute::is_void_v<StageCount_>,
              cutlass::gemm::collective::StageCountAutoCarveout<
                  static_cast<int>(
                      sizeof(typename CollectiveEpilogue::SharedStorage))>,
              StageCount_>,
          MainloopSchedule_>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
  using LayoutSFA = typename CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename CollectiveMainloop::InternalLayoutSFB;
  using BlockScaledConfig =
      typename CollectiveMainloop::Sm1xxBlkScaledConfig;

  template <class NewElementD>
  using RebindOutput = KernelConfig<
      TileM_, TileN_, TileK_, MainloopSchedule_, StageCount_, NewElementD,
      LayoutA_, EpilogueTile_, EpilogueSchedule_>;
};

using Kernel128x8x128 = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
using Kernel256x8x128 = KernelConfig<
    cute::_256, cute::_8, cute::_128,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
using Kernel256x8x64 = KernelConfig<
    cute::_256, cute::_8, cute::_64,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
using Kernel128x8x256 = KernelConfig<
    cute::_128, cute::_8, cute::_256,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
using Kernel128x8x256Stage3 = KernelConfig<
    cute::_128, cute::_8, cute::_256,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
    cutlass::gemm::collective::StageCount<3>>;
using Kernel64x8x256Pingpong = KernelConfig<
    cute::_64, cute::_8, cute::_256,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
    void, cutlass::half_t, cutlass::layout::RowMajor,
    cute::Shape<cute::_64, cute::_8>>;
using Kernel64x8x128CooperativeStage4 = KernelConfig<
    cute::_64, cute::_8, cute::_128,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
    cutlass::gemm::collective::StageCount<4>, cutlass::half_t,
    cutlass::layout::RowMajor, cute::Shape<cute::_64, cute::_8>,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized>;
using Kernel128x8x128Stage4 = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
    cutlass::gemm::collective::StageCount<4>>;
using Kernel128x8x128Stage6 = KernelConfig<
    cute::_128, cute::_8, cute::_128,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
    cutlass::gemm::collective::StageCount<6>>;
using Kernel128x8x64 = KernelConfig<
    cute::_128, cute::_8, cute::_64,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative, void,
    cutlass::half_t, cutlass::layout::ColumnMajor>;
using Kernel128x16x128Pingpong = KernelConfig<
    cute::_128, cute::_16, cute::_128,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>;
using Kernel128x32x128 = KernelConfig<
    cute::_128, cute::_32, cute::_128,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
using Kernel128x64x128 = KernelConfig<
    cute::_128, cute::_64, cute::_128,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
using Kernel128x128x128 = KernelConfig<
    cute::_128, cute::_128, cute::_128,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;

using ArrayProblemShape =
    cutlass::gemm::ArrayProblemShape<
        cute::Shape<int, int, int, int>>;

template <class ElementD_ = cutlass::half_t>
struct ArrayKernel128x8x128 {
  using ElementPairA = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
  using ElementPairB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementA = typename ElementPairA::DataType;
  using ElementB = typename ElementPairB::DataType;
  using ElementSF = typename ElementPairA::ScaleFactorType;
  using ElementC = void;
  using ElementD = ElementD_;
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
  using TileShape = cute::Shape<cute::_128, cute::_16, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD,
          cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>::
          CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementPairA, LayoutA, AlignmentA,
          ElementPairB, LayoutB, AlignmentB, ElementAccumulator, TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(
                  sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::
              KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ArrayProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
  using LayoutSFA = typename CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename CollectiveMainloop::InternalLayoutSFB;
  using BlockScaledConfig =
      typename CollectiveMainloop::Sm1xxBlkScaledConfig;

  template <class NewElementD>
  using RebindOutput = ArrayKernel128x8x128<NewElementD>;
};

}  // namespace mxfp6_gemm::grouped
