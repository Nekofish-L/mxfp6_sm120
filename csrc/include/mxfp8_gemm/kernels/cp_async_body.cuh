#pragma once
#include "mxfp8_gemm/kernels/async_pipeline.cuh"
#include "mxfp8_gemm/layout.cuh"
#include "mxfp8_gemm/pdl.cuh"
#include <cuda_bf16.h>
#include "cute/arch/mma_sm120.hpp"
#include "cute/arch/copy_sm75.hpp"

namespace mxfp8_cp_async {
using Mma = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,float,cutlass::float_ue8m0_t,32>;
using mxfp8_layout::row_major::sm_index;
using mxfp8_pipeline::prefetch;

// VectorReuse must fit its output tile inside pipeline storage. VectorSeparate
// uses the host descriptor's max(pipeline, output) allocation instead.
enum class Store { Scalar, VectorReuse, VectorSeparate };

} // namespace mxfp8_cp_async
