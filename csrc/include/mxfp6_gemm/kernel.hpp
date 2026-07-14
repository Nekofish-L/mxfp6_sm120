#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cute/tensor.hpp"

namespace mxfp6_gemm {

// OCP MXFP6 E3M2: 32 six-bit values share one UE8M0 power-of-two scale.
using ElementPairA = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
using ElementPairB = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
using ElementA = typename ElementPairA::DataType;
using ElementB = typename ElementPairB::DataType;
using ElementSF = typename ElementPairA::ScaleFactorType;

using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// FP6 TMA uses the 16U6 representation. 128 logical elements are 96 bytes,
// satisfying both the sub-byte packing and TMA alignment requirements.
inline constexpr int AlignmentA = 128;
inline constexpr int AlignmentB = 128;
inline constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
inline constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    cutlass::arch::OpClassTensorOp,
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
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
using BlockScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

static_assert(cutlass::sizeof_bits<ElementA>::value == 6);
static_assert(cutlass::sizeof_bits<ElementB>::value == 6);
static_assert(cutlass::sizeof_bits<ElementSF>::value == 8);

}  // namespace mxfp6_gemm
