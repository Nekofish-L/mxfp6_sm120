// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "cutlass/cuda_host_adapter.hpp"
#include "mxfp_common/pdl.cuh"

namespace mxfp8_dual_detail {

__device__ __forceinline__ int sf_index(int r, int g, int k) {
  return (r / 128) * (k / 128) * 512 + (g / 4) * 512
      + (r % 32) * 16 + ((r % 128) / 32) * 4 + g % 4;
}

template<int Rows>
__device__ __forceinline__ int sm_index(int r, int c) {
  return (c / 128) * Rows * 128 + r * 128 + ((c % 128) ^ ((r % 8) * 16));
}

inline CUtensorMap tensor_map(void* ptr, int rows, int tile_rows, int k, int bk) {
  CUtensorMap map{};
  uint64_t dimensions[3] = {128, uint64_t(rows), uint64_t(k / 128)};
  uint64_t strides[2] = {uint64_t(k), 128};
  uint32_t box[3] = {128, uint32_t(tile_rows), uint32_t(bk / 128)};
  uint32_t element_strides[3] = {1, 1, 1};
  CUresult status = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, ptr, dimensions, strides, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status == CUDA_SUCCESS, "TMA descriptor failed: ", int(status));
  return map;
}

} // namespace mxfp8_dual_detail
