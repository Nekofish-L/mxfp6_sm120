#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/library.h>

#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/util/packed_stride.hpp"
#include "mxfp6_gemm/kernel_grouped.hpp"
#include "mxfp6_gemm/quantization.hpp"

using namespace cute;

namespace mxfp6_gemm::torch_ext {

constexpr int kScaleRowsPerAtom = 128;
constexpr int kScaleGroupsPerAtom = 4;
constexpr int kScaleThreads = 256;
constexpr int kMxScaleVectorSize = 32;
constexpr int kQuantElementsPerThread = 8;
constexpr int kQuantThreadsPerGroup =
    kMxScaleVectorSize / kQuantElementsPerThread;
constexpr int kQuantGroupsPerBlock =
    kScaleThreads / kQuantThreadsPerGroup;
constexpr int kMaxMoEExperts = 257;
constexpr uint8_t kUe8m0One = 0x7f;

bool grouped_raster_along_n() {
  char const* value = std::getenv("MXFP6_GROUPED_RASTER_N");
  return value != nullptr && value[0] == '1';
}

int grouped_max_swizzle() {
  char const* value = std::getenv("MXFP6_GROUPED_SWIZZLE");
  return value == nullptr ? 1 : std::max(1, std::atoi(value));
}

bool grouped_fixed_tensormaps() {
  char const* value = std::getenv("MXFP6_GROUPED_FIXED_TMA");
  return value != nullptr && value[0] == '1';
}

bool grouped_explicit_tiles() {
  char const* value = std::getenv("MXFP6_GROUPED_EXPLICIT_TILES");
  return value != nullptr && value[0] == '1';
}

bool grouped_dual_cta_persistent() {
  char const* value = std::getenv("MXFP6_GROUPED_DUAL_CTA");
  return value != nullptr && value[0] == '1';
}

bool grouped_w1_wide(int64_t rows) {
  char const* value = std::getenv("MXFP6_GROUPED_W1_WIDE_THRESHOLD");
  if (value != nullptr) {
    return rows > std::max(0, std::atoi(value));
  }
  if (rows == 80) {
    return true;
  }
  // The 448-row wide launch leaves an expensive partial SM120 wave.
  return rows >= 384 && rows != 448;
}

bool grouped_reduce_x4() {
  char const* value = std::getenv("MXFP6_GROUPED_REDUCE_X4");
  return value == nullptr || value[0] != '0';
}

int grouped_w2_grid_m(int n, int64_t rows) {
  int const full_grid = (n + 127) / 128;
  char const* value = std::getenv("MXFP6_GROUPED_W2_GRID_M");
  if (value != nullptr) {
    return std::clamp(std::atoi(value), 1, full_grid);
  }
  return rows >= 256 && rows <= 320
      ? std::min(8, full_grid)
      : full_grid;
}

using Qwen35GroupedKernel =
    typename mxfp6_gemm::grouped::Kernel64x8x256Pingpong::
        template RebindOutput<cutlass::bfloat16_t>;
using Qwen35GroupedProblem =
    typename mxfp6_gemm::grouped::ProblemShape::UnderlyingProblemShape;

struct Qwen35GroupedMetadataDevice {
  uint8_t const* activation{};
  uint8_t const* activation_scales{};
  uint8_t const* weight{};
  uint8_t const* weight_scales{};
  int64_t const* expert_offsets{};
  int64_t const* scale_offsets{};
  typename Qwen35GroupedKernel::ElementD* output{};
  Qwen35GroupedProblem* problem_shapes{};
  typename Qwen35GroupedKernel::ElementA const** ptr_a{};
  typename Qwen35GroupedKernel::ElementB const** ptr_b{};
  typename Qwen35GroupedKernel::ElementSF const** ptr_sfa{};
  typename Qwen35GroupedKernel::ElementSF const** ptr_sfb{};
  typename Qwen35GroupedKernel::ElementD** ptr_d{};
  typename Qwen35GroupedKernel::StrideA* stride_a{};
  typename Qwen35GroupedKernel::StrideB* stride_b{};
  typename Qwen35GroupedKernel::StrideC* stride_c{};
  typename Qwen35GroupedKernel::StrideD* stride_d{};
  typename Qwen35GroupedKernel::LayoutSFA* layout_sfa{};
  typename Qwen35GroupedKernel::LayoutSFB* layout_sfb{};
  int32_t* work_tiles{};
  int32_t* work_tile_count{};
  int max_work_tiles{};
  int group_count{};
  int n{};
  int k{};

  __device__ __forceinline__ void write(int group) const {
    if (problem_shapes == nullptr || group >= group_count) {
      return;
    }
    int64_t const row_start = expert_offsets[group];
    int64_t const row_end = expert_offsets[group + 1];
    int const rows = static_cast<int>(row_end - row_start);
    auto problem = cute::make_shape(n, rows, k, 1);
    auto scale_problem = cute::make_shape(n, rows, max(k, 128), 1);
    int const packed_scale_groups =
        (k / 32 + kScaleGroupsPerAtom - 1) /
        kScaleGroupsPerAtom * kScaleGroupsPerAtom;

    problem_shapes[group] = cute::make_shape(n, rows, k);
    ptr_a[group] = reinterpret_cast<
        typename Qwen35GroupedKernel::ElementA const*>(
            weight + static_cast<int64_t>(group) * n * k * 3 / 4);
    ptr_b[group] = reinterpret_cast<
        typename Qwen35GroupedKernel::ElementB const*>(
            activation + row_start * k);
    ptr_sfa[group] = reinterpret_cast<
        typename Qwen35GroupedKernel::ElementSF const*>(
            weight_scales +
            static_cast<int64_t>(group) * n * packed_scale_groups);
    ptr_sfb[group] = reinterpret_cast<
        typename Qwen35GroupedKernel::ElementSF const*>(
            activation_scales +
            scale_offsets[group] * packed_scale_groups);
    ptr_d[group] = output + row_start * n;
    stride_a[group] = cutlass::make_cute_packed_stride(
        typename Qwen35GroupedKernel::StrideA{},
        cute::make_shape(n, k, 1));
    stride_b[group] = cutlass::make_cute_packed_stride(
        typename Qwen35GroupedKernel::StrideB{},
        cute::make_shape(rows, k, 1));
    stride_c[group] = cutlass::make_cute_packed_stride(
        typename Qwen35GroupedKernel::StrideC{},
        cute::make_shape(n, rows, 1));
    stride_d[group] = cutlass::make_cute_packed_stride(
        typename Qwen35GroupedKernel::StrideD{},
        cute::make_shape(n, rows, 1));
    layout_sfa[group] =
        Qwen35GroupedKernel::BlockScaledConfig::tile_atom_to_shape_SFA(
            scale_problem);
    layout_sfb[group] =
        Qwen35GroupedKernel::BlockScaledConfig::tile_atom_to_shape_SFB(
            scale_problem);
  }

  __device__ __forceinline__ void write_work_tiles(
      int group,
      int rows,
      int32_t* cursor) const {
    if (work_tiles == nullptr || group >= group_count || rows <= 0) {
      return;
    }
    constexpr int kTileM = 64;
    constexpr int kTileN = 8;
    int const tiles_m = (n + kTileM - 1) / kTileM;
    int const tiles_n = (rows + kTileN - 1) / kTileN;
    int const tile_count = tiles_m * tiles_n;
    int const base = atomicAdd(cursor, tile_count);
    for (int tile_n = 0; tile_n < tiles_n; ++tile_n) {
      for (int tile_m = 0; tile_m < tiles_m; ++tile_m) {
        int const tile = base + tile_n * tiles_m + tile_m;
        if (tile < max_work_tiles) {
          int32_t* entry = work_tiles + tile * 4;
          entry[0] = tile_m;
          entry[1] = tile_n;
          entry[2] = group;
          entry[3] = 1;
        }
      }
    }
  }
};

namespace qwen35_router {

constexpr int kHidden = 2048;
constexpr int kRoutedExperts = 256;
constexpr int kExpertsWithSharedGate = kRoutedExperts + 1;
constexpr int kTopK = 8;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kValuesPerThread = sizeof(uint4) / sizeof(__nv_bfloat16);
constexpr int kGroupsPerRow = kHidden / kMxScaleVectorSize;
constexpr int kQuantGroupsPerBlock = 8;
constexpr int kQuantBlocks =
    kGroupsPerRow / kQuantGroupsPerBlock;

template <int Tokens>
struct RouterSharedStorage {
  float warp_sums[Tokens][kWarps];
};


__device__ __forceinline__ void load_bfloat16x8(
    uint4 const& packed,
    float (&values)[kValuesPerThread]) {
  auto const* pairs =
      reinterpret_cast<__nv_bfloat162 const*>(&packed);
#pragma unroll
  for (int pair = 0; pair < kValuesPerThread / 2; ++pair) {
    float2 const converted = __bfloat1622float2(pairs[pair]);
    values[pair * 2] = converted.x;
    values[pair * 2 + 1] = converted.y;
  }
}

template <int Tokens, bool SeparateShared = false>
__global__ __launch_bounds__(kThreads, 2)
void router_quant_kernel(
    __nv_bfloat16 const* hidden,
    __nv_bfloat16 const* gate_weight,
    __nv_bfloat16 const* shared_gate_weight,
    uint8_t* quantized,
    uint8_t* logical_scales,
    __nv_bfloat16* routed_logits,
    float* topk_weights,
    int32_t* topk_ids,
    __nv_bfloat16* shared_gate,
    bool renormalize) {
  static_assert(
      Tokens == 1 || Tokens == 2 || Tokens == 4 || Tokens == 8 ||
      Tokens == 16);
  int const expert = static_cast<int>(blockIdx.x);
  int const thread = static_cast<int>(threadIdx.x);
  int const warp = thread / 32;
  int const lane = thread % 32;
  __shared__ RouterSharedStorage<Tokens> shared;

  // Reproduce quantize_mxfp8_logical's four-thread group reduction and
  // vector conversion exactly. Only eight router CTAs need this short
  // prologue; their remaining warps can start fetching router weights.
  if (expert < kQuantBlocks && warp == 0) {
    int const thread_in_group = lane % 4;
    int const group_in_warp = lane / 4;
    int const group =
        expert * kQuantGroupsPerBlock + group_in_warp;
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
      int const value_offset =
          token * kHidden + group * kMxScaleVectorSize +
          thread_in_group * kValuesPerThread;
      uint4 const packed =
          *reinterpret_cast<uint4 const*>(hidden + value_offset);
      float values[kValuesPerThread];
      load_bfloat16x8(packed, values);

      float absmax = 0.0f;
#pragma unroll
      for (int index = 0; index < kValuesPerThread; ++index) {
        absmax = fmaxf(absmax, fabsf(values[index]));
      }
      absmax = fmaxf(
          absmax,
          __shfl_xor_sync(0xffffffffu, absmax, 1, 4));
      absmax = fmaxf(
          absmax,
          __shfl_xor_sync(0xffffffffu, absmax, 2, 4));

      float inverse_scale = 1.0f;
      uint8_t scale_code = kUe8m0One;
      if (thread_in_group == 0) {
        float const raw_scale =
            fmaxf(absmax / 448.0f, 1.0e-30f);
        uint32_t scale_bits = __float_as_uint(raw_scale);
        scale_bits =
            (scale_bits + 0x007fffffu) & 0x7f800000u;
        scale_code =
            static_cast<uint8_t>(scale_bits >> 23);
        inverse_scale =
            1.0f / __uint_as_float(scale_bits);
        logical_scales[
            token * kGroupsPerRow + group] = scale_code;
      }
      inverse_scale = __shfl_sync(
          0xffffffffu, inverse_scale, 0, 4);

      uint16_t pairs[kValuesPerThread / 2];
#pragma unroll
      for (int index = 0;
           index < kValuesPerThread;
           index += 2) {
        pairs[index / 2] = __nv_cvt_float2_to_fp8x2(
            make_float2(
                values[index] * inverse_scale,
                values[index + 1] * inverse_scale),
            __NV_SATFINITE,
            __NV_E4M3);
      }
      uint2 const output{
          static_cast<uint32_t>(pairs[0]) |
              (static_cast<uint32_t>(pairs[1]) << 16),
          static_cast<uint32_t>(pairs[2]) |
              (static_cast<uint32_t>(pairs[3]) << 16)};
      *reinterpret_cast<uint2*>(
          quantized + value_offset) = output;
    }
  }

  float accumulators[Tokens]{};
  __nv_bfloat16 const* expert_weight =
      SeparateShared && expert == kRoutedExperts
      ? shared_gate_weight
      : gate_weight + static_cast<int64_t>(expert) * kHidden;
  constexpr int kValuesPerIteration =
      kThreads * kValuesPerThread;
  constexpr int kIterations = kHidden / kValuesPerIteration;
#pragma unroll
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    int const column =
        iteration * kValuesPerIteration +
        thread * kValuesPerThread;
    uint4 const weight_packed =
        *reinterpret_cast<uint4 const*>(
            expert_weight + column);
    float weight_values[kValuesPerThread];
    load_bfloat16x8(weight_packed, weight_values);
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
      uint4 const hidden_packed =
          *reinterpret_cast<uint4 const*>(
              hidden + token * kHidden + column);
      float hidden_values[kValuesPerThread];
      load_bfloat16x8(hidden_packed, hidden_values);
#pragma unroll
      for (int value = 0;
           value < kValuesPerThread;
           ++value) {
        accumulators[token] = fmaf(
            hidden_values[value],
            weight_values[value],
            accumulators[token]);
      }
    }
  }

// Match the four-warp FP32 reduction used by vLLM's small-batch router GEMV.
#pragma unroll
  for (int token = 0; token < Tokens; ++token) {
    float sum = accumulators[token];
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    if (lane == 0) {
      shared.warp_sums[token][warp] = sum;
    }
  }
  __syncthreads();

  if (thread == 0) {
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
      float sum = 0.0f;
#pragma unroll
      for (int source_warp = 0;
           source_warp < kWarps;
           ++source_warp) {
        sum += shared.warp_sums[token][source_warp];
      }
      __nv_bfloat16 const logit = __float2bfloat16(sum);
      if (expert < kRoutedExperts) {
        routed_logits[
            token * kRoutedExperts + expert] = logit;
      } else {
        shared_gate[token] = logit;
      }
    }
  }

  cooperative_groups::this_grid().sync();
  if (thread == 0) {
    cutlass::arch::launch_dependent_grids();
  }

  // For each token, one warp exactly mirrors vLLM's 256-expert BF16
  // topkGating specialization: eight values per lane, butterfly reductions,
  // and smaller expert IDs winning ties.
  if (expert >= Tokens || warp != 0) {
    return;
  }
  int const token = expert;
  float probabilities[8];
  int expert_ids[8];
#pragma unroll
  for (int load = 0; load < 4; ++load) {
    int const column = lane * 2 + load * 64;
    probabilities[load * 2] = __bfloat162float(
        routed_logits[token * kRoutedExperts + column]);
    probabilities[load * 2 + 1] = __bfloat162float(
        routed_logits[token * kRoutedExperts + column + 1]);
    expert_ids[load * 2] = column;
    expert_ids[load * 2 + 1] = column + 1;
  }

  float row_max = probabilities[0];
#pragma unroll
  for (int value = 1; value < 8; ++value) {
    row_max = fmaxf(row_max, probabilities[value]);
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    row_max = fmaxf(
        row_max,
        __shfl_xor_sync(0xffffffffu, row_max, mask));
  }

  float row_sum = 0.0f;
#pragma unroll
  for (int value = 0; value < 8; ++value) {
    probabilities[value] =
        expf(probabilities[value] - row_max);
    row_sum += probabilities[value];
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    row_sum += __shfl_xor_sync(
        0xffffffffu, row_sum, mask);
  }
  float const inverse_sum = 1.0f / row_sum;
#pragma unroll
  for (int value = 0; value < 8; ++value) {
    probabilities[value] *= inverse_sum;
    if (isnan(probabilities[value]) ||
        isinf(probabilities[value])) {
      probabilities[value] = 0.0f;
    }
  }

#pragma unroll
  for (int choice = 0; choice < kTopK; ++choice) {
    float candidate = probabilities[0];
    int candidate_expert = expert_ids[0];
#pragma unroll
    for (int value = 1; value < 8; ++value) {
      if (probabilities[value] > candidate) {
        candidate = probabilities[value];
        candidate_expert = expert_ids[value];
      }
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      float const other_candidate = __shfl_xor_sync(
          0xffffffffu, candidate, mask);
      int const other_expert = __shfl_xor_sync(
          0xffffffffu, candidate_expert, mask);
      if (other_candidate > candidate ||
          (other_candidate == candidate &&
           other_expert < candidate_expert)) {
        candidate = other_candidate;
        candidate_expert = other_expert;
      }
    }
    if (lane == 0) {
      topk_weights[token * kTopK + choice] = candidate;
      topk_ids[token * kTopK + choice] = candidate_expert;
    }
    if (choice + 1 < kTopK) {
#pragma unroll
      for (int value = 0; value < 8; ++value) {
        if (expert_ids[value] == candidate_expert) {
          probabilities[value] = -10000.0f;
        }
      }
    }
  }
  if (lane == 0 && renormalize) {
    float selected_sum = 0.0f;
#pragma unroll
    for (int choice = 0; choice < kTopK; ++choice) {
      selected_sum +=
          topk_weights[token * kTopK + choice];
    }
    float const inverse_selected_sum =
        selected_sum > 0.0f ? 1.0f / selected_sum : 0.0f;
#pragma unroll
    for (int choice = 0; choice < kTopK; ++choice) {
      topk_weights[token * kTopK + choice] *=
          inverse_selected_sum;
    }
  }
}


template <int Tokens, bool SeparateShared = false>
void launch(
    at::Tensor& quantized,
    at::Tensor& logical_scales,
    at::Tensor& routed_logits,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    at::Tensor& shared_gate,
    at::Tensor const& hidden,
    at::Tensor const& gate_weight,
    at::Tensor const* shared_gate_weight,
    bool renormalize,
    cudaStream_t stream) {
  int active_blocks = 0;
  C10_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          router_quant_kernel<Tokens, SeparateShared>,
          kThreads,
          0));
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(hidden.get_device());
  TORCH_CHECK(
      active_blocks * properties.multiProcessorCount >=
          kExpertsWithSharedGate,
      "Qwen3.5 fused router requires at least ",
      kExpertsWithSharedGate,
      " simultaneously resident CTAs; device supports ",
      active_blocks * properties.multiProcessorCount);

  auto const* hidden_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(
          hidden.data_ptr());
  auto const* gate_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(
          gate_weight.data_ptr());
  __nv_bfloat16 const* shared_gate_weight_ptr = nullptr;
  if constexpr (SeparateShared) {
    shared_gate_weight_ptr =
        reinterpret_cast<__nv_bfloat16 const*>(
            shared_gate_weight->data_ptr());
  }
  auto* quantized_ptr = quantized.data_ptr<uint8_t>();
  auto* scales_ptr = logical_scales.data_ptr<uint8_t>();
  auto* logits_ptr =
      reinterpret_cast<__nv_bfloat16*>(
          routed_logits.data_ptr());
  auto* weights_ptr = topk_weights.data_ptr<float>();
  auto* ids_ptr = topk_ids.data_ptr<int32_t>();
  auto* shared_gate_ptr =
      reinterpret_cast<__nv_bfloat16*>(
          shared_gate.data_ptr());
  void* arguments[] = {
      &hidden_ptr,
      &gate_ptr,
      &shared_gate_weight_ptr,
      &quantized_ptr,
      &scales_ptr,
      &logits_ptr,
      &weights_ptr,
      &ids_ptr,
      &shared_gate_ptr,
      &renormalize};
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          router_quant_kernel<Tokens, SeparateShared>),
      dim3(kExpertsWithSharedGate),
      dim3(kThreads),
      arguments,
      0,
      stream));
}

__device__ __forceinline__ int64_t dense_scale_offset(
    int row,
    int k_group) {
  constexpr int kTilesPerRow =
      (kGroupsPerRow + 3) / 4;
  int64_t const tile =
      static_cast<int64_t>(row / 128) * kTilesPerRow +
      k_group / 4;
  int const row_in_tile = row % 128;
  return tile * 128 * 4 +
      (row_in_tile % 32) * 16 +
      (row_in_tile / 32) * 4 +
      k_group % 4;
}

__device__ __forceinline__ int64_t grouped_scale_offset(
    int64_t packed_row,
    int k_group) {
  constexpr int kGroupsPerAtom = 4;
  constexpr int kTilesPerRow =
      (kGroupsPerRow + kGroupsPerAtom - 1) /
      kGroupsPerAtom;
  int64_t const tile =
      (packed_row / 128) * kTilesPerRow +
      k_group / kGroupsPerAtom;
  int const row_in_tile =
      static_cast<int>(packed_row % 128);
  return tile * 128 * kGroupsPerAtom +
      (row_in_tile % 32) * 16 +
      (row_in_tile / 32) * 4 +
      k_group % kGroupsPerAtom;
}

template <bool Route>
__global__ __launch_bounds__(kThreads, 2)
void topk_quant_kernel(
    __nv_bfloat16 const* hidden,
    __nv_bfloat16 const* routed_logits,
    uint8_t* quantized,
    uint8_t* logical_scales,
    uint8_t* packed_scales,
    float* topk_weights,
    int32_t* topk_ids,
    int tokens,
    bool renormalize,
    uint8_t* permuted_activation,
    uint8_t* grouped_scales,
    int32_t* expert_cursors,
    int64_t* expert_offsets,
    int64_t* scale_offsets,
    int32_t* inverse_permutation,
    int32_t* source_tokens,
    Qwen35GroupedMetadataDevice grouped_metadata) {
  int const token = static_cast<int>(blockIdx.x);
  int const thread = static_cast<int>(threadIdx.x);
  int const warp = thread / 32;
  int const lane = thread % 32;
  int const thread_in_group = thread % 4;
  int const group = thread / 4;
  int const value_offset =
      token * kHidden +
      group * kMxScaleVectorSize +
      thread_in_group * kValuesPerThread;
  uint4 const packed =
      *reinterpret_cast<uint4 const*>(hidden + value_offset);
  float values[kValuesPerThread];
  load_bfloat16x8(packed, values);

  float absmax = 0.0f;
#pragma unroll
  for (int index = 0; index < kValuesPerThread; ++index) {
    absmax = fmaxf(absmax, fabsf(values[index]));
  }
  absmax = fmaxf(
      absmax,
      __shfl_xor_sync(0xffffffffu, absmax, 1, 4));
  absmax = fmaxf(
      absmax,
      __shfl_xor_sync(0xffffffffu, absmax, 2, 4));

  float inverse_scale = 1.0f;
  if (thread_in_group == 0) {
    float const raw_scale =
        fmaxf(absmax / 448.0f, 1.0e-30f);
    uint32_t scale_bits = __float_as_uint(raw_scale);
    scale_bits =
        (scale_bits + 0x007fffffu) & 0x7f800000u;
    uint8_t const scale_code =
        static_cast<uint8_t>(scale_bits >> 23);
    inverse_scale =
        1.0f / __uint_as_float(scale_bits);
    logical_scales[
        token * kGroupsPerRow + group] = scale_code;
    packed_scales[dense_scale_offset(token, group)] =
        scale_code;
  }
  inverse_scale = __shfl_sync(
      0xffffffffu, inverse_scale, 0, 4);

  uint16_t pairs[kValuesPerThread / 2];
#pragma unroll
  for (int index = 0;
       index < kValuesPerThread;
       index += 2) {
    pairs[index / 2] = __nv_cvt_float2_to_fp8x2(
        make_float2(
            values[index] * inverse_scale,
            values[index + 1] * inverse_scale),
        __NV_SATFINITE,
        __NV_E4M3);
  }
  uint2 const output{
      static_cast<uint32_t>(pairs[0]) |
          (static_cast<uint32_t>(pairs[1]) << 16),
      static_cast<uint32_t>(pairs[2]) |
          (static_cast<uint32_t>(pairs[3]) << 16)};
  *reinterpret_cast<uint2*>(
      quantized + value_offset) = output;

  if (warp == 0) {
    float probabilities[8];
    int expert_ids[8];
#pragma unroll
    for (int load = 0; load < 4; ++load) {
      int const column = lane * 2 + load * 64;
      probabilities[load * 2] = __bfloat162float(
          routed_logits[token * kRoutedExperts + column]);
      probabilities[load * 2 + 1] = __bfloat162float(
          routed_logits[
              token * kRoutedExperts + column + 1]);
      expert_ids[load * 2] = column;
      expert_ids[load * 2 + 1] = column + 1;
    }

    float row_max = probabilities[0];
#pragma unroll
    for (int value = 1; value < 8; ++value) {
      row_max = fmaxf(row_max, probabilities[value]);
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      row_max = fmaxf(
          row_max,
          __shfl_xor_sync(
              0xffffffffu, row_max, mask));
    }

    float row_sum = 0.0f;
#pragma unroll
    for (int value = 0; value < 8; ++value) {
      probabilities[value] =
          expf(probabilities[value] - row_max);
      row_sum += probabilities[value];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      row_sum += __shfl_xor_sync(
          0xffffffffu, row_sum, mask);
    }
    float const inverse_sum = 1.0f / row_sum;
#pragma unroll
    for (int value = 0; value < 8; ++value) {
      probabilities[value] *= inverse_sum;
      if (isnan(probabilities[value]) ||
          isinf(probabilities[value])) {
        probabilities[value] = 0.0f;
      }
    }

#pragma unroll
    for (int choice = 0; choice < kTopK; ++choice) {
      float candidate = probabilities[0];
      int candidate_expert = expert_ids[0];
#pragma unroll
      for (int value = 1; value < 8; ++value) {
        if (probabilities[value] > candidate) {
          candidate = probabilities[value];
          candidate_expert = expert_ids[value];
        }
      }
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        float const other_candidate =
            __shfl_xor_sync(
                0xffffffffu, candidate, mask);
        int const other_expert =
            __shfl_xor_sync(
                0xffffffffu, candidate_expert, mask);
        if (other_candidate > candidate ||
            (other_candidate == candidate &&
             other_expert < candidate_expert)) {
          candidate = other_candidate;
          candidate_expert = other_expert;
        }
      }
      if (lane == 0) {
        topk_weights[
            token * kTopK + choice] = candidate;
        topk_ids[token * kTopK + choice] =
            candidate_expert;
      }
      if (choice + 1 < kTopK) {
#pragma unroll
        for (int value = 0; value < 8; ++value) {
          if (expert_ids[value] == candidate_expert) {
            probabilities[value] = -10000.0f;
          }
        }
      }
    }
    if (lane == 0 && renormalize) {
      float selected_sum = 0.0f;
#pragma unroll
      for (int choice = 0; choice < kTopK; ++choice) {
        selected_sum += topk_weights[token * kTopK + choice];
      }
      float const inverse_selected_sum =
          selected_sum > 0.0f ? 1.0f / selected_sum : 0.0f;
#pragma unroll
      for (int choice = 0; choice < kTopK; ++choice) {
        topk_weights[token * kTopK + choice] *=
            inverse_selected_sum;
      }
    }
  }

  if constexpr (Route) {
    cooperative_groups::this_grid().sync();
    __shared__ int32_t expert_histogram[kRoutedExperts];
    __shared__ int64_t warp_row_prefix[kWarps];
    __shared__ int64_t warp_scale_prefix[kWarps];
    __shared__ int32_t work_tile_cursor;
    if (token == 0) {
      expert_histogram[thread] = 0;
      if (thread == 0) {
        work_tile_cursor = 0;
      }
      __syncthreads();
      for (int route = thread;
           route < tokens * kTopK;
           route += kThreads) {
        atomicAdd(
            expert_histogram + topk_ids[route], 1);
      }
      __syncthreads();
      int64_t const rows = expert_histogram[thread];
      int64_t row_prefix = rows;
      int64_t const padded_rows =
          (rows + 127) & -static_cast<int64_t>(128);
      int64_t scale_prefix = padded_rows;
#pragma unroll
      for (int offset = 1; offset < 32; offset *= 2) {
        int64_t const row_add = __shfl_up_sync(
            0xffffffffu, row_prefix, offset);
        int64_t const scale_add = __shfl_up_sync(
            0xffffffffu, scale_prefix, offset);
        if (lane >= offset) {
          row_prefix += row_add;
          scale_prefix += scale_add;
        }
      }
      if (lane == 31) {
        warp_row_prefix[warp] = row_prefix;
        warp_scale_prefix[warp] = scale_prefix;
      }
      __syncthreads();
      if (warp == 0) {
        int64_t warp_rows =
            lane < kWarps ? warp_row_prefix[lane] : 0;
        int64_t warp_scale_rows =
            lane < kWarps ? warp_scale_prefix[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
          int64_t const row_add = __shfl_up_sync(
              0xffffffffu, warp_rows, offset);
          int64_t const scale_add = __shfl_up_sync(
              0xffffffffu, warp_scale_rows, offset);
          if (lane >= offset) {
            warp_rows += row_add;
            warp_scale_rows += scale_add;
          }
        }
        if (lane < kWarps) {
          warp_row_prefix[lane] = warp_rows;
          warp_scale_prefix[lane] = warp_scale_rows;
        }
      }
      __syncthreads();
      int64_t const prior_rows =
          warp == 0 ? 0 : warp_row_prefix[warp - 1];
      int64_t const prior_scale_rows =
          warp == 0 ? 0 : warp_scale_prefix[warp - 1];
      int64_t const row_offset =
          prior_rows + row_prefix - rows;
      int64_t const scale_offset =
          prior_scale_rows + scale_prefix - padded_rows;
      expert_offsets[thread] = row_offset;
      scale_offsets[thread] = scale_offset;
      expert_cursors[thread] =
          static_cast<int32_t>(row_offset);
      grouped_metadata.write_work_tiles(
          thread,
          static_cast<int>(rows),
          &work_tile_cursor);
      if (thread == kRoutedExperts - 1) {
        expert_offsets[kRoutedExperts] =
            prior_rows + row_prefix;
        scale_offsets[kRoutedExperts] =
            prior_scale_rows + scale_prefix;
      }
      __syncthreads();
      if (thread == 0 && grouped_metadata.work_tile_count != nullptr) {
        *grouped_metadata.work_tile_count = work_tile_cursor;
      }
    }
    cooperative_groups::this_grid().sync();
    if (token == 0) {
      grouped_metadata.write(thread);
    }

    int const route_index = token * kTopK + warp;
    int const destination_expert =
        topk_ids[route_index];
    int destination = 0;
    if (lane == 0) {
      destination = atomicAdd(
          expert_cursors + destination_expert,
          1);
      inverse_permutation[route_index] =
          destination;
    }
    destination = __shfl_sync(
        0xffffffffu, destination, 0);
    if (source_tokens != nullptr) {
      if (lane == 0) {
        source_tokens[destination] = token;
      }
      __syncthreads();
      if (thread == 0) {
        cutlass::arch::launch_dependent_grids();
      }
      return;
    }
    auto const* source_values = reinterpret_cast<uint4 const*>(
        quantized + static_cast<int64_t>(token) * kHidden);
    auto* destination_values = reinterpret_cast<uint4*>(
        permuted_activation +
        static_cast<int64_t>(destination) * kHidden);
#pragma unroll
    for (int vector = lane;
         vector < kHidden / static_cast<int>(sizeof(uint4));
         vector += 32) {
      destination_values[vector] =
          source_values[vector];
    }
    int64_t const local_row =
        destination -
        expert_offsets[destination_expert];
    int64_t const packed_row =
        scale_offsets[destination_expert] +
        local_row;
#pragma unroll
    for (int scale_group = lane;
         scale_group < kGroupsPerRow;
         scale_group += 32) {
      grouped_scales[grouped_scale_offset(
          packed_row, scale_group)] =
          logical_scales[
              token * kGroupsPerRow + scale_group];
    }
    __syncthreads();
    if (thread == 0) {
      cutlass::arch::launch_dependent_grids();
    }
  }
}

}  // namespace qwen35_router

namespace direct_moe {

using ReferenceKernel = mxfp6_gemm::grouped::Kernel128x8x128;
using ReferenceMainloop = typename ReferenceKernel::CollectiveMainloop;
using TiledMma = typename ReferenceMainloop::TiledMma;
using MmaOp = typename TiledMma::MMA_Op;
using MmaTraits = cute::MMA_Traits<MmaOp>;
using SmemCopyAtomA = typename ReferenceMainloop::SmemCopyAtomA;
using SmemCopyAtomB = typename ReferenceMainloop::SmemCopyAtomB;

using SwapElementPairA =
    cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using SwapElementPairB =
    cutlass::mx_float6_t<cutlass::float_e3m2_t>;
using SwapTileShape =
    cute::Shape<cute::_16, cute::_8, cute::_128>;
using SwapClusterShape =
    cute::Shape<cute::_1, cute::_1, cute::_1>;
using SwapMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        SwapElementPairA,
        cutlass::layout::RowMajor,
        128,
        SwapElementPairB,
        cutlass::layout::ColumnMajor,
        128,
        float,
        SwapTileShape,
        SwapClusterShape,
        cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::
            KernelTmaWarpSpecializedMxf8f6f4Sm120>::CollectiveOp;
using SwapTiledMma = typename SwapMainloop::TiledMma;
using SwapMmaOp = typename SwapTiledMma::MMA_Op;
using SwapSmemCopyAtomA = typename SwapMainloop::SmemCopyAtomA;
using SwapSmemCopyAtomB = typename SwapMainloop::SmemCopyAtomB;

using ReduceElementPairA =
    cutlass::mx_float6_t<cutlass::float_e3m2_t>;
using ReduceElementPairB =
    cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ReduceTileShape =
    cute::Shape<cute::_16, cute::_8, cute::_128>;
using ReduceMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        ReduceElementPairA,
        cutlass::layout::RowMajor,
        128,
        ReduceElementPairB,
        cutlass::layout::ColumnMajor,
        128,
        float,
        ReduceTileShape,
        SwapClusterShape,
        cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::
            KernelTmaWarpSpecializedMxf8f6f4Sm120>::CollectiveOp;
using ReduceTiledMma = typename ReduceMainloop::TiledMma;
using ReduceMmaOp = typename ReduceTiledMma::MMA_Op;
using ReduceSmemCopyAtomA = typename ReduceMainloop::SmemCopyAtomA;
using ReduceSmemCopyAtomB = typename ReduceMainloop::SmemCopyAtomB;

using FusedNormalElementPairA =
    cutlass::mx_float6_t<cutlass::float_e3m2_t>;
using FusedNormalElementPairB =
    cutlass::mx_float8_t<cutlass::float_e4m3_t>;

using GroupedFusedTileShape =
    cute::Shape<cute::_128, cute::_8, cute::_256>;
using GroupedFusedMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        FusedNormalElementPairA,
        cutlass::layout::RowMajor,
        128,
        FusedNormalElementPairB,
        cutlass::layout::ColumnMajor,
        128,
        float,
        GroupedFusedTileShape,
        SwapClusterShape,
        cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::
            KernelTmaWarpSpecializedMxf8f6f4Sm120>::CollectiveOp;
using GroupedFusedTiledMma =
    typename GroupedFusedMainloop::TiledMma;
using GroupedFusedMmaOp =
    typename GroupedFusedTiledMma::MMA_Op;
using GroupedFusedSmemCopyAtomA =
    typename GroupedFusedMainloop::SmemCopyAtomA;
using GroupedFusedSmemCopyAtomB =
    typename GroupedFusedMainloop::SmemCopyAtomB;

using GroupedFusedNarrowMainloop =
    typename Qwen35GroupedKernel::CollectiveMainloop;
using GroupedFusedNarrowTiledMma =
    typename GroupedFusedNarrowMainloop::TiledMma;
using GroupedFusedNarrowMmaOp =
    typename GroupedFusedNarrowTiledMma::MMA_Op;
using GroupedFusedNarrowSmemCopyAtomA =
    typename GroupedFusedNarrowMainloop::SmemCopyAtomA;
using GroupedFusedNarrowSmemCopyAtomB =
    typename GroupedFusedNarrowMainloop::SmemCopyAtomB;

template <bool Wide>
using SelectedGroupedFusedMainloop = std::conditional_t<
    Wide, GroupedFusedMainloop, GroupedFusedNarrowMainloop>;

using QwenW1TileShape =
    cute::Shape<cute::_128, cute::_8, cute::_256>;
using QwenW1Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        FusedNormalElementPairA,
        cutlass::layout::RowMajor,
        128,
        FusedNormalElementPairB,
        cutlass::layout::ColumnMajor,
        128,
        float,
        QwenW1TileShape,
        SwapClusterShape,
        cutlass::gemm::collective::StageCount<2>,
        cutlass::gemm::
            KernelTmaWarpSpecializedMxf8f6f4Sm120>::CollectiveOp;
using QwenW1TiledMma = typename QwenW1Mainloop::TiledMma;
using QwenW1MmaOp = typename QwenW1TiledMma::MMA_Op;
using QwenW1SmemCopyAtomA =
    typename QwenW1Mainloop::SmemCopyAtomA;
using QwenW1SmemCopyAtomB =
    typename QwenW1Mainloop::SmemCopyAtomB;

using ReferenceSmemLayoutA =
    typename ReferenceMainloop::SmemLayoutA;
using ReferenceSmemLayoutB =
    typename ReferenceMainloop::SmemLayoutB;
using SmemLayoutA = decltype(
    ReferenceSmemLayoutA{}(_, _, cute::Int<0>{}));
using SmemLayoutB = decltype(
    ReferenceSmemLayoutB{}(_, _, cute::Int<0>{}));
using ReferenceSwapSmemLayoutA =
    typename SwapMainloop::SmemLayoutA;
using ReferenceSwapSmemLayoutB =
    typename SwapMainloop::SmemLayoutB;
using SwapSmemLayoutA = decltype(
    ReferenceSwapSmemLayoutA{}(_, _, cute::Int<0>{}));
using SwapSmemLayoutB = decltype(
    ReferenceSwapSmemLayoutB{}(_, _, cute::Int<0>{}));
using ReferenceReduceSmemLayoutA =
    typename ReduceMainloop::SmemLayoutA;
using ReferenceReduceSmemLayoutB =
    typename ReduceMainloop::SmemLayoutB;
using ReduceSmemLayoutA = decltype(
    ReferenceReduceSmemLayoutA{}(_, _, cute::Int<0>{}));
using ReduceSmemLayoutB = decltype(
    ReferenceReduceSmemLayoutB{}(_, _, cute::Int<0>{}));
using ReferenceGroupedFusedSmemLayoutA =
    typename GroupedFusedMainloop::SmemLayoutA;
using ReferenceGroupedFusedSmemLayoutB =
    typename GroupedFusedMainloop::SmemLayoutB;
using GroupedFusedSmemLayoutA = decltype(
    ReferenceGroupedFusedSmemLayoutA{}(
        _, _, cute::Int<0>{}));
using GroupedFusedSmemLayoutB = decltype(
    ReferenceGroupedFusedSmemLayoutB{}(
        _, _, cute::Int<0>{}));
using ReferenceGroupedFusedNarrowSmemLayoutA =
    typename GroupedFusedNarrowMainloop::SmemLayoutA;
using ReferenceGroupedFusedNarrowSmemLayoutB =
    typename GroupedFusedNarrowMainloop::SmemLayoutB;
using GroupedFusedNarrowSmemLayoutA = decltype(
    ReferenceGroupedFusedNarrowSmemLayoutA{}(
        _, _, cute::Int<0>{}));
using GroupedFusedNarrowSmemLayoutB = decltype(
    ReferenceGroupedFusedNarrowSmemLayoutB{}(
        _, _, cute::Int<0>{}));
using ReferenceQwenW1SmemLayoutA =
    typename QwenW1Mainloop::SmemLayoutA;
using ReferenceQwenW1SmemLayoutB =
    typename QwenW1Mainloop::SmemLayoutB;
using QwenW1SmemLayoutA = decltype(
    ReferenceQwenW1SmemLayoutA{}(
        _, _, cute::Int<0>{}));
using QwenW1SmemLayoutB = decltype(
    ReferenceQwenW1SmemLayoutB{}(
        _, _, cute::Int<0>{}));

static constexpr int kTileM = 128;
static constexpr int kTileN = 8;
static constexpr int kTileK = 128;
static constexpr int kQwenW1TileK = 256;
static constexpr int kThreads = 256;
static constexpr int kSwapTileM = 16;
static constexpr int kSwapWarpN = 8;
static constexpr int kSwapWarps = 4;
static constexpr int kSwapTileN = kSwapWarpN * kSwapWarps;
static constexpr int kSwapThreads = 32 * kSwapWarps;
static constexpr int kFusedSwapWarps = 8;
static constexpr int kFusedSwapThreads = 32 * kFusedSwapWarps;
static constexpr int kFusedSwapPairsPerWarp = 4;
static constexpr int kFusedSwapPairsPerBlock =
    kFusedSwapPairsPerWarp * kFusedSwapWarps;
static constexpr int kGroupedFusedTileM = 128;
static constexpr int kGroupedFusedTileK = 256;
static constexpr int kGroupedFusedThreads = 256;
static constexpr int kGroupedFusedPairsPerBlock =
    kGroupedFusedTileM / 2;
static constexpr int kGroupedFusedNarrowTileM = 64;
static constexpr int kGroupedFusedNarrowThreads = 128;
static constexpr int kGroupedFusedNarrowPairsPerBlock =
    kGroupedFusedNarrowTileM / 2;
static constexpr int kReduceTileM = 16;
static constexpr int kReduceMaxRoutes = 9;
static constexpr int kReduceThreads = 32 * kReduceMaxRoutes;
static constexpr int kSwapRouteThreshold = 16;
static constexpr int kSwapMinK = 1024;
static constexpr int kFp6ValuesPerSegment = 16;
static constexpr int kFp6BytesPerSegment = 12;
static constexpr int kFp6SharedBytesPerSegment = 16;

struct alignas(1024) SharedStorage {
  alignas(1024) uint8_t a[cute::cosize_v<SmemLayoutA>];
  alignas(1024) uint8_t b[cute::cosize_v<SmemLayoutB>];
};

struct alignas(1024) SwapSharedStorage {
  alignas(1024) uint8_t a[cute::cosize_v<SwapSmemLayoutA>];
  alignas(1024) uint8_t
      b[kSwapWarps][cute::cosize_v<SwapSmemLayoutB>];
};

struct alignas(1024) FusedSwapSharedStorage {
  alignas(1024) uint8_t a[cute::cosize_v<SwapSmemLayoutA>];
  alignas(1024) uint8_t
      b[kFusedSwapWarps][cute::cosize_v<SwapSmemLayoutB>];
  float warp_absmax[kFusedSwapWarps];
  float inverse_scale;
  uint8_t scale_code;
};

struct alignas(1024) GroupedFusedSharedStorage {
  alignas(1024) uint8_t
      a[cute::cosize_v<GroupedFusedSmemLayoutA>];
  alignas(1024) uint8_t
      b[cute::cosize_v<GroupedFusedSmemLayoutB>];
  float accumulators[kTileN][kGroupedFusedTileM];
  int32_t source_rows[2][kTileN];
  alignas(16) uint64_t tma_barrier;
};

struct alignas(1024) GroupedFusedNarrowSharedStorage {
  alignas(1024) uint8_t
      a[cute::cosize_v<GroupedFusedNarrowSmemLayoutA>];
  alignas(1024) uint8_t
      b[cute::cosize_v<GroupedFusedNarrowSmemLayoutB>];
  float accumulators[kTileN][kGroupedFusedNarrowTileM];
  int32_t source_rows[2][kTileN];
  alignas(16) uint64_t tma_barrier;
};

struct alignas(1024) SplitKSharedStorage {
  alignas(1024) uint8_t
      a[cute::cosize_v<QwenW1SmemLayoutA>];
  alignas(1024) uint8_t
      b[cute::cosize_v<QwenW1SmemLayoutB>];
  float inverse_scale[2];
  uint8_t scale_code[2];
};

struct alignas(1024) ReduceSharedStorage {
  alignas(1024) uint8_t
      a[kReduceMaxRoutes][cute::cosize_v<ReduceSmemLayoutA>];
  alignas(1024) uint8_t
      b[kReduceMaxRoutes][cute::cosize_v<ReduceSmemLayoutB>];
  float partial[kReduceMaxRoutes][kReduceTileM];
};

__device__ __forceinline__ int64_t scale_offset(
    int row,
    int k_block,
    int k_blocks) {
  int64_t const row_tile = row / kScaleRowsPerAtom;
  int const row_in_tile = row % kScaleRowsPerAtom;
  int const k_tiles =
      (k_blocks + kScaleGroupsPerAtom - 1) / kScaleGroupsPerAtom;
  int64_t const tile =
      row_tile * k_tiles + k_block / kScaleGroupsPerAtom;
  return tile * kScaleRowsPerAtom * kScaleGroupsPerAtom +
      (row_in_tile % 32) * 16 +
      (row_in_tile / 32) * 4 +
      k_block % kScaleGroupsPerAtom;
}

template <int SourceBytes>
__device__ __forceinline__ void copy_global_to_shared_16(
    void* destination,
    void const* source) {
  static_assert(SourceBytes == 12 || SourceBytes == 16);
  uint32_t const shared_address = static_cast<uint32_t>(
      __cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
      :
      : "r"(shared_address), "l"(source), "n"(SourceBytes));
}

__device__ __forceinline__ void copy_global_to_shared_4(
    void* destination,
    void const* source) {
  uint32_t const shared_address = static_cast<uint32_t>(
      __cvta_generic_to_shared(destination));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 4;\n"
      :
      : "r"(shared_address), "l"(source));
}

__device__ __forceinline__ void copy_fp6_segment_to_shared(
    uint8_t* destination,
    uint8_t const* source) {
  copy_global_to_shared_4(destination, source);
  copy_global_to_shared_4(destination + 4, source + 4);
  copy_global_to_shared_4(destination + 8, source + 8);
  *reinterpret_cast<uint32_t*>(destination + 12) = 0;
}

__device__ __forceinline__ void commit_global_to_shared() {
  asm volatile("cp.async.commit_group;\n" : :);
}

__device__ __forceinline__ void wait_global_to_shared() {
  asm volatile("cp.async.wait_group 0;\n" : :);
}

template <class Id, class ElementD>
__global__ __launch_bounds__(kThreads, 2)
void route_w6a8_kernel(
    uint8_t const* activation,
    uint8_t const* logical_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    Id const* topk_ids,
    ElementD* output,
    int activation_rows,
    int routed_topk,
    int include_shared,
    int num_experts,
    int n,
    int k) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  int const route = blockIdx.y;
  int const routes_per_token = routed_topk + include_shared;
  int const token = route / routes_per_token;
  int const route_lane = route - token * routes_per_token;
  int const expert =
      include_shared && route_lane == routed_topk
      ? num_experts - 1
      : static_cast<int>(
            topk_ids[static_cast<int64_t>(token) * routed_topk +
                     route_lane]);
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  int const source_row =
      activation_rows == gridDim.y ? route : token;
  int const tile_m = blockIdx.x * kTileM;
  int const thread = threadIdx.x;
  int const warp = thread / 32;
  int const lane = thread % 32;
  int const k_blocks = k / kMxScaleVectorSize;
  int const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;

  uint8_t const* expert_weight =
      weight + static_cast<int64_t>(expert) * n * k * 3 / 4;
  uint8_t const* expert_scales =
      weight_scales +
      static_cast<int64_t>(expert) * n * packed_k_blocks;
  uint8_t const* source_activation =
      activation + static_cast<int64_t>(source_row) * k;
  uint8_t const* source_scales =
      logical_scales +
      static_cast<int64_t>(source_row) * k_blocks;

  extern __shared__ char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<SharedStorage*>(dynamic_shared);
  auto s_a = cute::make_tensor(
      cute::make_smem_ptr(shared.a), SmemLayoutA{});
  auto s_b = cute::make_tensor(
      cute::make_smem_ptr(shared.b), SmemLayoutB{});

  TiledMma tiled_mma;
  ReferenceMainloop reference_mainloop;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto fragment_a = thread_mma.partition_fragment_A(s_a);
  auto fragment_b = thread_mma.partition_fragment_B(s_b);
  auto tiled_copy_a =
      cute::make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
  auto tiled_copy_b =
      cute::make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
  auto thread_copy_a = tiled_copy_a.get_thread_slice(thread);
  auto thread_copy_b = tiled_copy_b.get_thread_slice(thread);
  auto source_a = thread_copy_a.partition_S(
      cute::as_position_independent_swizzle_tensor(s_a));
  auto source_b = thread_copy_b.partition_S(
      cute::as_position_independent_swizzle_tensor(s_b));
  auto register_a = thread_copy_a.retile_D(fragment_a);
  auto register_b = thread_copy_b.retile_D(fragment_b);

  float accum0 = 0.0f;
  float accum1 = 0.0f;
  float accum2 = 0.0f;
  float accum3 = 0.0f;

  auto const sfa_layout =
      reference_mainloop.get_layoutSFA_TV(tiled_mma);
  int const sfa_row = sfa_layout(thread, 0) % kTileM;
  for (int k_tile = 0; k_tile < k; k_tile += kTileK) {
    constexpr int segments_per_row =
        kTileK / kFp6ValuesPerSegment;
    for (int segment = thread;
         segment < kTileM * segments_per_row;
         segment += kThreads) {
      int const row = segment / segments_per_row;
      int const segment_in_row = segment % segments_per_row;
      int const source_segment =
          segment_in_row ^ (row & 7);
      uint8_t const* source =
          expert_weight +
          static_cast<int64_t>(tile_m + row) * k * 3 / 4 +
          static_cast<int64_t>(k_tile) * 3 / 4 +
          source_segment * kFp6BytesPerSegment;
      int const destination_offset = s_a.layout()(
          row, segment_in_row * kFp6SharedBytesPerSegment);
      uint8_t* destination = shared.a + destination_offset;
      copy_fp6_segment_to_shared(
          destination, source);
    }

    constexpr int kActivationSegments = kTileK / sizeof(uint4);
    for (int segment = thread;
         segment < kTileN * kActivationSegments;
         segment += kThreads) {
      int const row = segment / kActivationSegments;
      int const column =
          segment % kActivationSegments * sizeof(uint4);
      int const destination_offset = s_b.layout()(row, column);
      if (row == 0) {
        copy_global_to_shared_16<16>(
            shared.b + destination_offset,
            source_activation + k_tile + column);
      } else {
        *reinterpret_cast<uint4*>(shared.b + destination_offset) =
            uint4{};
      }
    }
    commit_global_to_shared();
    wait_global_to_shared();
    __syncthreads();

    CUTE_UNROLL
    for (int k_block = 0; k_block < kTileK / 32; ++k_block) {
      cute::copy(
          tiled_copy_a,
          source_a(_, _, k_block),
          register_a(_, _, k_block));
      cute::copy(
          tiled_copy_b,
          source_b(_, _, k_block),
          register_b(_, _, k_block));

      auto a_words =
          cute::recast<uint32_t>(fragment_a(_, _, k_block));
      auto b_words =
          cute::recast<uint32_t>(fragment_b(_, _, k_block));
      int const global_k_block = k_tile / 32 + k_block;
      uint8_t const sfa = expert_scales[scale_offset(
          tile_m + sfa_row,
          global_k_block,
          k_blocks)];
      uint8_t const block_sfb = source_scales[global_k_block];
      MmaOp::fma(
          accum0, accum1, accum2, accum3,
          a_words(0), a_words(1), a_words(2), a_words(3),
          b_words(0), b_words(1),
          accum0, accum1, accum2, accum3,
          sfa, block_sfb);
    }
    __syncthreads();
  }

  float const accumulators[4] = {
      accum0, accum1, accum2, accum3};
  if ((lane & 3) == 0) {
    CUTE_UNROLL
    for (int value = 0; value < 4; value += 2) {
      int const local_m = lane / 4 + value / 2 * 8;
      int const output_column = tile_m + warp * 16 + local_m;
      output[static_cast<int64_t>(route) * n + output_column] =
          static_cast<ElementD>(accumulators[value]);
    }
  }
#else
  CUTE_INVALID_CONTROL_PATH(
      "direct MXFP6 MoE requires the sm_120a target");
#endif
}

template <class ElementD, bool UsePdl>
__global__ __launch_bounds__(kThreads, 4)
void grouped_w6a8_static_kernel(
    uint8_t const* activation,
    uint8_t const* activation_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    int64_t const* expert_offsets,
    int64_t const* scale_offsets,
    ElementD* output,
    int num_experts,
    int n,
    int k) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  using StaticMainloop = ReferenceMainloop;
  using StaticTiledMma = TiledMma;
  using StaticMmaOp = MmaOp;
  using StaticSmemCopyAtomA = SmemCopyAtomA;
  using StaticSmemCopyAtomB = SmemCopyAtomB;
  using StaticSmemLayoutA = SmemLayoutA;
  using StaticSmemLayoutB = SmemLayoutB;
  constexpr int kStaticTileK = kTileK;
  if constexpr (UsePdl) {
    cutlass::arch::wait_on_dependent_grids();
  }
  int const expert = static_cast<int>(blockIdx.y);
  int64_t const row_start = expert_offsets[expert];
  int const rows = static_cast<int>(
      expert_offsets[expert + 1] - row_start);
  if (expert >= num_experts || rows <= 0) {
    if constexpr (UsePdl) {
      cutlass::arch::launch_dependent_grids();
    }
    return;
  }

  int const thread = static_cast<int>(threadIdx.x);
  int const k_blocks = k / kMxScaleVectorSize;
  int const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  uint8_t const* expert_weight =
      weight + static_cast<int64_t>(expert) * n * k * 3 / 4;
  uint8_t const* expert_scales =
      weight_scales +
      static_cast<int64_t>(expert) * n * packed_k_blocks;

  extern __shared__ char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<SharedStorage*>(dynamic_shared);
  auto s_a = cute::make_tensor(
      cute::make_smem_ptr(shared.a), StaticSmemLayoutA{});
  auto s_b = cute::make_tensor(
      cute::make_smem_ptr(shared.b), StaticSmemLayoutB{});
  StaticTiledMma tiled_mma;
  StaticMainloop reference_mainloop;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto fragment_a = thread_mma.partition_fragment_A(s_a);
  auto fragment_b = thread_mma.partition_fragment_B(s_b);
  auto tiled_copy_a =
      cute::make_tiled_copy_A(StaticSmemCopyAtomA{}, tiled_mma);
  auto tiled_copy_b =
      cute::make_tiled_copy_B(StaticSmemCopyAtomB{}, tiled_mma);
  auto thread_copy_a = tiled_copy_a.get_thread_slice(thread);
  auto thread_copy_b = tiled_copy_b.get_thread_slice(thread);
  auto source_a = thread_copy_a.partition_S(
      cute::as_position_independent_swizzle_tensor(s_a));
  auto source_b = thread_copy_b.partition_S(
      cute::as_position_independent_swizzle_tensor(s_b));
  auto register_a = thread_copy_a.retile_D(fragment_a);
  auto register_b = thread_copy_b.retile_D(fragment_b);
  auto const sfa_layout =
      reference_mainloop.get_layoutSFA_TV(tiled_mma);
  auto const sfb_layout =
      reference_mainloop.get_layoutSFB_TV(tiled_mma);
  int const sfa_row = sfa_layout(thread, 0) % kTileM;
  int const sfb_row = sfb_layout(thread, 0) % kTileN;
  auto c_identity = cute::make_identity_tensor(
      cute::Shape<cute::_128, cute::_8>{});
  auto c_coordinates = cute::composition(
      c_identity, tiled_mma.get_layoutC_TV());

  for (int tile_m = static_cast<int>(blockIdx.x) * kTileM;
       tile_m < n;
       tile_m += static_cast<int>(gridDim.x) * kTileM) {
    for (int row_base = 0; row_base < rows; row_base += kTileN) {
      float accum0 = 0.0f;
      float accum1 = 0.0f;
      float accum2 = 0.0f;
      float accum3 = 0.0f;
      for (int k_tile = 0; k_tile < k; k_tile += kStaticTileK) {
        constexpr int kSegmentsPerRow =
            kStaticTileK / kFp6ValuesPerSegment;
        for (int segment = thread;
             segment < kTileM * kSegmentsPerRow;
             segment += kThreads) {
          int const row = segment / kSegmentsPerRow;
          int const segment_in_row = segment % kSegmentsPerRow;
          int const source_segment =
              segment_in_row ^ (row & 7);
          uint8_t const* source =
              expert_weight +
              static_cast<int64_t>(tile_m + row) * k * 3 / 4 +
              static_cast<int64_t>(k_tile) * 3 / 4 +
              source_segment * kFp6BytesPerSegment;
          int const destination_offset = s_a.layout()(
              row,
              segment_in_row * kFp6SharedBytesPerSegment);
          copy_fp6_segment_to_shared(
              shared.a + destination_offset, source);
        }
        constexpr int kActivationSegments =
            kStaticTileK / sizeof(uint4);
        for (int segment = thread;
             segment < kTileN * kActivationSegments;
             segment += kThreads) {
          int const row = segment / kActivationSegments;
          int const segment_in_row = segment % kActivationSegments;
          int const column = segment_in_row * sizeof(uint4);
          int const source_column =
              (segment_in_row ^ (row & 7)) * sizeof(uint4);
          int const local_row = row_base + row;
          int const destination_offset = s_b.layout()(row, column);
          if (local_row < rows) {
            copy_global_to_shared_16<16>(
                shared.b + destination_offset,
                activation +
                    (row_start + local_row) * k +
                    k_tile + source_column);
          } else {
            *reinterpret_cast<uint4*>(
                shared.b + destination_offset) = uint4{};
          }
        }
        commit_global_to_shared();
        int const first_global_k_block =
            k_tile / kMxScaleVectorSize;
        uint32_t const sfa_codes =
            *reinterpret_cast<uint32_t const*>(
                expert_scales + scale_offset(
                    tile_m + sfa_row,
                    first_global_k_block,
                    k_blocks));
        int const local_scale_row = row_base + sfb_row;
        uint32_t sfb_codes = 0x7f7f7f7fu;
        if (local_scale_row < rows) {
          sfb_codes = *reinterpret_cast<uint32_t const*>(
              activation_scales + scale_offset(
                  scale_offsets[expert] + local_scale_row,
                  first_global_k_block,
                  k_blocks));
        }
        wait_global_to_shared();
        __syncthreads();

        CUTE_UNROLL
        for (int k_block = 0;
             k_block < kStaticTileK / kMxScaleVectorSize;
             ++k_block) {
          cute::copy(
              tiled_copy_a,
              source_a(_, _, k_block),
              register_a(_, _, k_block));
          cute::copy(
              tiled_copy_b,
              source_b(_, _, k_block),
              register_b(_, _, k_block));
          auto a_words =
              cute::recast<uint32_t>(fragment_a(_, _, k_block));
          auto b_words =
              cute::recast<uint32_t>(fragment_b(_, _, k_block));
          int const scale_shift = (k_block & 3) * 8;
          uint8_t const sfa = static_cast<uint8_t>(
              sfa_codes >> scale_shift);
          uint8_t const sfb = static_cast<uint8_t>(
              sfb_codes >> scale_shift);
          StaticMmaOp::fma(
              accum0, accum1, accum2, accum3,
              a_words(0), a_words(1), a_words(2), a_words(3),
              b_words(0), b_words(1),
              accum0, accum1, accum2, accum3,
              sfa, sfb);
        }
        __syncthreads();
      }

      float const accumulators[4] = {
          accum0, accum1, accum2, accum3};
      CUTE_UNROLL
      for (int value = 0; value < 4; ++value) {
        auto const coordinate = c_coordinates(thread, value);
        int const output_column =
            tile_m + static_cast<int>(cute::get<0>(coordinate));
        int const output_row =
            row_base + static_cast<int>(cute::get<1>(coordinate));
        if (output_row < rows && output_column < n) {
          output[(row_start + output_row) * n + output_column] =
              static_cast<ElementD>(accumulators[value]);
        }
      }
    }
  }
  if constexpr (UsePdl) {
    cutlass::arch::launch_dependent_grids();
  }
#else
  CUTE_INVALID_CONTROL_PATH(
      "static grouped MXFP6 MoE requires the sm_120a target");
#endif
}

template <class Id, class ElementD>
__global__ __launch_bounds__(kSwapThreads, 4)
void route_w6a8_swapab_kernel(
    uint8_t const* activation,
    uint8_t const* logical_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    Id const* topk_ids,
    ElementD* output,
    int activation_rows,
    int routed_topk,
    int include_shared,
    int num_experts,
    int n,
    int k) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  int const route = blockIdx.y;
  int const routes_per_token = routed_topk + include_shared;
  int const token = route / routes_per_token;
  int const route_lane = route - token * routes_per_token;
  int const expert =
      include_shared && route_lane == routed_topk
      ? num_experts - 1
      : static_cast<int>(
            topk_ids[static_cast<int64_t>(token) * routed_topk +
                     route_lane]);
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  int const source_row =
      activation_rows == gridDim.y ? route : token;
  int const tile_n = blockIdx.x * kSwapTileN;
  int const thread = threadIdx.x;
  int const warp = thread / 32;
  int const lane = thread % 32;
  int const k_blocks = k / kMxScaleVectorSize;
  int const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;

  uint8_t const* expert_weight =
      weight + static_cast<int64_t>(expert) * n * k * 3 / 4;
  uint8_t const* expert_scales =
      weight_scales +
      static_cast<int64_t>(expert) * n * packed_k_blocks;
  uint8_t const* source_activation =
      activation + static_cast<int64_t>(source_row) * k;
  uint8_t const* source_scales =
      logical_scales +
      static_cast<int64_t>(source_row) * k_blocks;

  extern __shared__ char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<SwapSharedStorage*>(dynamic_shared);
  auto s_a = cute::make_tensor(
      cute::make_smem_ptr(shared.a), SwapSmemLayoutA{});
  auto s_b = cute::make_tensor(
      cute::make_smem_ptr(shared.b[warp]), SwapSmemLayoutB{});

  SwapTiledMma tiled_mma;
  SwapMainloop reference_mainloop;
  auto thread_mma = tiled_mma.get_thread_slice(lane);
  auto fragment_a = thread_mma.partition_fragment_A(s_a);
  auto fragment_b = thread_mma.partition_fragment_B(s_b);
  auto tiled_copy_a =
      cute::make_tiled_copy_A(SwapSmemCopyAtomA{}, tiled_mma);
  auto tiled_copy_b =
      cute::make_tiled_copy_B(SwapSmemCopyAtomB{}, tiled_mma);
  auto thread_copy_a = tiled_copy_a.get_thread_slice(lane);
  auto thread_copy_b = tiled_copy_b.get_thread_slice(lane);
  auto source_a = thread_copy_a.partition_S(
      cute::as_position_independent_swizzle_tensor(s_a));
  auto source_b = thread_copy_b.partition_S(
      cute::as_position_independent_swizzle_tensor(s_b));
  auto register_a = thread_copy_a.retile_D(fragment_a);
  auto register_b = thread_copy_b.retile_D(fragment_b);

  for (int linear = thread;
       linear < kSwapTileM * kTileK;
       linear += kSwapThreads) {
    s_a(linear / kTileK, linear % kTileK) = 0;
  }
  __syncthreads();

  auto const sfb_layout =
      reference_mainloop.get_layoutSFB_TV(tiled_mma);
  int const sfb_row = sfb_layout(lane, 0) % kSwapWarpN;
  float accum0 = 0.0f;
  float accum1 = 0.0f;
  float accum2 = 0.0f;
  float accum3 = 0.0f;

  for (int k_tile = 0; k_tile < k; k_tile += kTileK) {
    for (int column = thread;
         column < kTileK;
         column += kSwapThreads) {
      s_a(0, column) = source_activation[k_tile + column];
    }

    constexpr int segments_per_row =
        kTileK / kFp6ValuesPerSegment;
    for (int segment = thread;
         segment < kSwapWarps * kSwapWarpN * segments_per_row;
         segment += kSwapThreads) {
      constexpr int segments_per_warp =
          kSwapWarpN * segments_per_row;
      int const destination_warp = segment / segments_per_warp;
      int const local_segment = segment % segments_per_warp;
      int const row = local_segment / segments_per_row;
      int const segment_in_row =
          local_segment % segments_per_row;
      int const source_segment =
          segment_in_row ^ (row & 7);
      uint8_t const* source =
          expert_weight +
          static_cast<int64_t>(
              tile_n + destination_warp * kSwapWarpN + row) *
              k * 3 / 4 +
          static_cast<int64_t>(k_tile) * 3 / 4 +
          source_segment * kFp6BytesPerSegment;
      auto const* source_words =
          reinterpret_cast<uint32_t const*>(source);
      uint4 const packed{
          source_words[0], source_words[1], source_words[2], 0};
      int const destination_offset = s_b.layout()(
          row, segment_in_row * kFp6SharedBytesPerSegment);
      *reinterpret_cast<uint4*>(
          shared.b[destination_warp] + destination_offset) = packed;
    }
    __syncthreads();

    CUTE_UNROLL
    for (int k_block = 0; k_block < kTileK / 32; ++k_block) {
      cute::copy(
          tiled_copy_a,
          source_a(_, _, k_block),
          register_a(_, _, k_block));
      cute::copy(
          tiled_copy_b,
          source_b(_, _, k_block),
          register_b(_, _, k_block));

      auto a_words =
          cute::recast<uint32_t>(fragment_a(_, _, k_block));
      auto b_words =
          cute::recast<uint32_t>(fragment_b(_, _, k_block));
      int const global_k_block = k_tile / 32 + k_block;
      uint8_t const block_sfa = source_scales[global_k_block];
      uint8_t const sfb = expert_scales[scale_offset(
          tile_n + warp * kSwapWarpN + sfb_row,
          global_k_block,
          k_blocks)];
      SwapMmaOp::fma(
          accum0, accum1, accum2, accum3,
          a_words(0), a_words(1), a_words(2), a_words(3),
          b_words(0), b_words(1),
          accum0, accum1, accum2, accum3,
          block_sfa, sfb);
    }
    __syncthreads();
  }

  if (lane < 4) {
    int const output_column =
        tile_n + warp * kSwapWarpN + lane * 2;
    output[static_cast<int64_t>(route) * n + output_column] =
        static_cast<ElementD>(accum0);
    output[static_cast<int64_t>(route) * n + output_column + 1] =
        static_cast<ElementD>(accum1);
  }
#else
  CUTE_INVALID_CONTROL_PATH(
      "direct swapAB MXFP6 MoE requires the sm_120a target");
#endif
}

__device__ __forceinline__ float round_bfloat16(float value) {
  return static_cast<float>(static_cast<cutlass::bfloat16_t>(value));
}

__device__ __forceinline__ uint16_t quantize_e4m3_pair(
    float first,
    float second) {
  return __nv_cvt_float2_to_fp8x2(
      make_float2(first, second), __NV_SATFINITE, __NV_E4M3);
}

template <
    int Tokens,
    int Splits = 4 / Tokens,
    bool IncludeShared = true,
    bool SeparateShared = false,
    bool ClusterReduce = false>
__global__ __launch_bounds__(kThreads, 1)
void qwen35_w1_splitk_kernel(
    uint8_t const* activation,
    uint8_t const* logical_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    uint8_t const* shared_weight,
    uint8_t const* shared_weight_scales,
    int32_t const* topk_ids,
    float* partial,
    uint8_t* output,
    uint8_t* output_scales) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  cutlass::arch::wait_on_dependent_grids();
  constexpr int kQwenK = 2048;
  constexpr int kQwenIntermediate = 256;
  constexpr int kQwenGateUp = 2 * kQwenIntermediate;
  constexpr int kRoutesPerToken = IncludeShared ? 9 : 8;
  constexpr int kQwenRoutes = Tokens * kRoutesPerToken;
  constexpr int kPairsPerBlock = kTileM / 2;
  constexpr int kQwenPairBlocks =
      kQwenIntermediate / kPairsPerBlock;
  static_assert(
      Tokens == 1 || Tokens == 2 || Tokens == 4 || Tokens == 8);
  static_assert(Splits == 1 || Splits == 2 || Splits == 4);
  static_assert(!ClusterReduce || Splits == 2);
  constexpr int kSplits = Splits;
  constexpr int kSplitK = kQwenK / kSplits;
  constexpr int kWorkItems =
      kQwenRoutes * kQwenPairBlocks * kSplits;
  static_assert(
      kWorkItems == 128 || kWorkItems == 144 ||
      kWorkItems == 256 || kWorkItems == 288);

  int const work = blockIdx.x;
  int const split = work % kSplits;
  int const pair_group =
      (work / kSplits) % kQwenPairBlocks;
  int const route =
      work / (kSplits * kQwenPairBlocks);
  int const token = route / kRoutesPerToken;
  int const route_lane = route % kRoutesPerToken;
  bool const shared_route = IncludeShared && route_lane == 8;
  int expert = shared_route
      ? 256
      : topk_ids[token * 8 + route_lane];
  constexpr int kNumExperts =
      IncludeShared && !SeparateShared ? 257 : 256;
  if (expert < 0 || expert >= kNumExperts) {
    expert = 0;
  }
  int const pair_block =
      pair_group * kPairsPerBlock;
  int const thread = threadIdx.x;
  int const warp = thread / 32;
  int const lane = thread % 32;
  constexpr int kBlocks = kQwenK / kMxScaleVectorSize;

  uint8_t const* expert_weight =
      SeparateShared && shared_route
      ? shared_weight
      : weight +
            static_cast<int64_t>(expert) *
                kQwenGateUp * kQwenK * 3 / 4;
  uint8_t const* expert_scales =
      SeparateShared && shared_route
      ? shared_weight_scales
      : weight_scales +
            static_cast<int64_t>(expert) *
                kQwenGateUp * kBlocks;
  uint8_t const* source_activation =
      activation + token * kQwenK;
  uint8_t const* source_scales =
      logical_scales + token * kBlocks;

  extern __shared__ char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<SplitKSharedStorage*>(
          dynamic_shared);
  auto s_a = cute::make_tensor(
      cute::make_smem_ptr(shared.a),
      QwenW1SmemLayoutA{});
  auto s_b = cute::make_tensor(
      cute::make_smem_ptr(shared.b),
      QwenW1SmemLayoutB{});
  QwenW1TiledMma tiled_mma;
  QwenW1Mainloop reference_mainloop;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto fragment_a = thread_mma.partition_fragment_A(s_a);
  auto fragment_b = thread_mma.partition_fragment_B(s_b);
  auto tiled_copy_a = cute::make_tiled_copy_A(
      QwenW1SmemCopyAtomA{}, tiled_mma);
  auto tiled_copy_b = cute::make_tiled_copy_B(
      QwenW1SmemCopyAtomB{}, tiled_mma);
  auto thread_copy_a =
      tiled_copy_a.get_thread_slice(thread);
  auto thread_copy_b =
      tiled_copy_b.get_thread_slice(thread);
  auto source_a = thread_copy_a.partition_S(
      cute::as_position_independent_swizzle_tensor(s_a));
  auto source_b = thread_copy_b.partition_S(
      cute::as_position_independent_swizzle_tensor(s_b));
  auto register_a = thread_copy_a.retile_D(fragment_a);
  auto register_b = thread_copy_b.retile_D(fragment_b);

  for (int linear = thread;
       linear < kTileN * kQwenW1TileK;
       linear += kThreads) {
    s_b(
        linear / kQwenW1TileK,
        linear % kQwenW1TileK) = 0;
  }
  __syncthreads();

  auto const sfa_layout =
      reference_mainloop.get_layoutSFA_TV(tiled_mma);
  int const local_sfa_row =
      sfa_layout(thread, 0) % kTileM;
  int const sfa_row =
      pair_block +
      (local_sfa_row < kPairsPerBlock
           ? local_sfa_row
           : kQwenIntermediate + local_sfa_row -
                 kPairsPerBlock);
  float accum0 = 0.0f;
  float accum1 = 0.0f;
  float accum2 = 0.0f;
  float accum3 = 0.0f;

  int const k_begin = split * kSplitK;
  int const k_end = k_begin + kSplitK;
  for (int k_tile = k_begin;
       k_tile < k_end;
       k_tile += kQwenW1TileK) {
    if (thread < kQwenW1TileK) {
      s_b(0, thread) =
          source_activation[k_tile + thread];
    }
    constexpr int segments_per_row =
        kQwenW1TileK / kFp6ValuesPerSegment;
    for (int segment = thread;
         segment <
             kTileM * segments_per_row;
         segment += kThreads) {
      int const local_row = segment / segments_per_row;
      int const segment_in_row =
          segment % segments_per_row;
      int const weight_row =
          pair_block +
          (local_row < kPairsPerBlock
               ? local_row
               : kQwenIntermediate + local_row -
                     kPairsPerBlock);
      int const source_segment =
          segment_in_row ^ (local_row & 7);
      uint8_t const* source =
          expert_weight +
          static_cast<int64_t>(weight_row) *
              kQwenK * 3 / 4 +
          static_cast<int64_t>(k_tile) * 3 / 4 +
          source_segment * kFp6BytesPerSegment;
      auto const* source_words =
          reinterpret_cast<uint32_t const*>(source);
      uint4 const packed{
          source_words[0],
          source_words[1],
          source_words[2],
          0};
      int const destination_offset = s_a.layout()(
          local_row,
          segment_in_row * kFp6SharedBytesPerSegment);
      *reinterpret_cast<uint4*>(
          shared.a + destination_offset) = packed;
    }
    int const first_global_k_block =
        k_tile / kMxScaleVectorSize;
    uint32_t const sfa_codes0 =
        *reinterpret_cast<uint32_t const*>(
            expert_scales + scale_offset(
                sfa_row, first_global_k_block, kBlocks));
    uint32_t const sfa_codes1 =
        *reinterpret_cast<uint32_t const*>(
            expert_scales + scale_offset(
                sfa_row, first_global_k_block + 4, kBlocks));
    uint32_t const sfb_codes0 =
        *reinterpret_cast<uint32_t const*>(
            source_scales + first_global_k_block);
    uint32_t const sfb_codes1 =
        *reinterpret_cast<uint32_t const*>(
            source_scales + first_global_k_block + 4);
    __syncthreads();

    CUTE_UNROLL
    for (int k_block = 0;
         k_block < kQwenW1TileK / kMxScaleVectorSize;
         ++k_block) {
      cute::copy(
          tiled_copy_a,
          source_a(_, _, k_block),
          register_a(_, _, k_block));
      cute::copy(
          tiled_copy_b,
          source_b(_, _, k_block),
          register_b(_, _, k_block));
      auto a_words =
          cute::recast<uint32_t>(
              fragment_a(_, _, k_block));
      auto b_words =
          cute::recast<uint32_t>(
              fragment_b(_, _, k_block));
      uint32_t const sfa_codes =
          k_block < 4 ? sfa_codes0 : sfa_codes1;
      uint32_t const sfb_codes =
          k_block < 4 ? sfb_codes0 : sfb_codes1;
      int const scale_shift = (k_block & 3) * 8;
      uint8_t const sfa = static_cast<uint8_t>(
          sfa_codes >> scale_shift);
      uint8_t const sfb = static_cast<uint8_t>(
          sfb_codes >> scale_shift);
      QwenW1MmaOp::fma(
          accum0,
          accum1,
          accum2,
          accum3,
          a_words(0),
          a_words(1),
          a_words(2),
          a_words(3),
          b_words(0),
          b_words(1),
          accum0,
          accum1,
          accum2,
          accum3,
          sfa,
          sfb);
    }
    __syncthreads();
  }

  float gate0 = 0.0f;
  float gate1 = 0.0f;
  float up0 = 0.0f;
  float up1 = 0.0f;
  int final_work = work;
  if constexpr (ClusterReduce) {
    // B=4 has two K-split CTAs per route/output group. Pair them in a
    // two-block cluster so the first CTA can consume the peer accumulator
    // from distributed shared memory instead of waiting at a grid-wide
    // barrier and round-tripping both halves through global memory.
    float* local_partial = reinterpret_cast<float*>(shared.a);
    if ((lane & 3) == 0) {
      int const row0 = warp * 16 + lane / 4;
      int const row1 = row0 + 8;
      local_partial[row0] = accum0;
      local_partial[row1] = accum2;
    }
    auto cluster = cooperative_groups::this_cluster();
    cluster.sync();
    if (split == 0 && thread < kPairsPerBlock / 2) {
      float const* peer_partial =
          cluster.map_shared_rank(local_partial, 1);
      int const local_pair = thread * 2;
      int const up_offset = kPairsPerBlock;
      gate0 =
          local_partial[local_pair] + peer_partial[local_pair];
      gate1 =
          local_partial[local_pair + 1] +
          peer_partial[local_pair + 1];
      up0 =
          local_partial[up_offset + local_pair] +
          peer_partial[up_offset + local_pair];
      up1 =
          local_partial[up_offset + local_pair + 1] +
          peer_partial[up_offset + local_pair + 1];
    }
    // Keep the peer CTA alive until all remote shared-memory reads finish.
    cluster.sync();
    if (thread == 0) {
      cutlass::arch::launch_dependent_grids();
    }
    if (split != 0) {
      return;
    }
    final_work = work / kSplits;
  } else {
    if ((lane & 3) == 0) {
      int const row0 = warp * 16 + lane / 4;
      int const row1 = row0 + 8;
      partial[
          static_cast<int64_t>(work) *
              kTileM +
          row0] = accum0;
      partial[
          static_cast<int64_t>(work) *
              kTileM +
          row1] = accum2;
    }

    cooperative_groups::this_grid().sync();
    if (thread == 0) {
      cutlass::arch::launch_dependent_grids();
    }
  }

  constexpr int kFinalBlocks =
      kQwenRoutes * kQwenPairBlocks;
  if (final_work >= kFinalBlocks) {
    return;
  }
  int const final_route = final_work / kQwenPairBlocks;
  int const final_group = final_work % kQwenPairBlocks;
  int const first_partial_work =
      (final_route * kQwenPairBlocks + final_group) *
      kSplits;
  float value0 = 0.0f;
  float value1 = 0.0f;
  if (thread < kPairsPerBlock / 2) {
    int const local_pair = thread * 2;
    int64_t const partial0 =
        static_cast<int64_t>(first_partial_work) *
        kTileM;
    int const up_offset = kPairsPerBlock;
    if constexpr (!ClusterReduce && kSplits == 1) {
      gate0 = partial[partial0 + local_pair];
      gate1 = partial[partial0 + local_pair + 1];
      up0 = partial[partial0 + up_offset + local_pair];
      up1 = partial[partial0 + up_offset + local_pair + 1];
    } else if constexpr (!ClusterReduce && kSplits == 2) {
      int64_t const partial1 =
          partial0 + kTileM;
      gate0 =
          partial[partial0 + local_pair] +
          partial[partial1 + local_pair];
      gate1 =
          partial[partial0 + local_pair + 1] +
          partial[partial1 + local_pair + 1];
      up0 =
          partial[partial0 + up_offset + local_pair] +
          partial[partial1 + up_offset + local_pair];
      up1 =
          partial[
              partial0 + up_offset + local_pair + 1] +
          partial[
              partial1 + up_offset + local_pair + 1];
    } else if constexpr (!ClusterReduce) {
      int64_t const partial1 =
          partial0 + kTileM;
      int64_t const partial2 =
          partial1 + kTileM;
      int64_t const partial3 =
          partial2 + kTileM;
      gate0 =
          ((partial[partial0 + local_pair] +
            partial[partial1 + local_pair]) +
           partial[partial2 + local_pair]) +
          partial[partial3 + local_pair];
      gate1 =
          ((partial[partial0 + local_pair + 1] +
            partial[partial1 + local_pair + 1]) +
           partial[partial2 + local_pair + 1]) +
          partial[partial3 + local_pair + 1];
      up0 =
          ((partial[
                partial0 + up_offset + local_pair] +
            partial[
                partial1 + up_offset + local_pair]) +
           partial[
               partial2 + up_offset + local_pair]) +
          partial[
              partial3 + up_offset + local_pair];
      up1 =
          ((partial[
                partial0 + up_offset + local_pair + 1] +
            partial[
                partial1 + up_offset + local_pair + 1]) +
           partial[
               partial2 + up_offset + local_pair + 1]) +
          partial[
              partial3 + up_offset + local_pair + 1];
    }
    gate0 = round_bfloat16(gate0);
    gate1 = round_bfloat16(gate1);
    up0 = round_bfloat16(up0);
    up1 = round_bfloat16(up1);
    gate0 = round_bfloat16(
        gate0 / (1.0f + expf(-gate0)));
    gate1 = round_bfloat16(
        gate1 / (1.0f + expf(-gate1)));
    value0 = round_bfloat16(gate0 * up0);
    value1 = round_bfloat16(gate1 * up1);
  }

  if (warp == 0) {
    float absmax =
        fmaxf(fabsf(value0), fabsf(value1));
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
      absmax = fmaxf(
          absmax,
          __shfl_down_sync(
              0xffffffffu, absmax, offset, 16));
    }
    if ((lane & 15) == 0) {
      int const scale_group = lane / 16;
      float const raw_scale =
          fmaxf(absmax / 448.0f, 1.0e-30f);
      uint32_t scale_bits = __float_as_uint(raw_scale);
      scale_bits =
          (scale_bits + 0x007fffffu) & 0x7f800000u;
      shared.scale_code[scale_group] =
          static_cast<uint8_t>(scale_bits >> 23);
      shared.inverse_scale[scale_group] =
          1.0f / __uint_as_float(scale_bits);
      output_scales[
          final_route *
              (kQwenIntermediate / kMxScaleVectorSize) +
          final_group * 2 + scale_group] =
              shared.scale_code[scale_group];
    }
  }
  __syncthreads();

  if (thread < kPairsPerBlock / 2) {
    int const scale_group = thread / 16;
    uint16_t const pair = quantize_e4m3_pair(
        value0 * shared.inverse_scale[scale_group],
        value1 * shared.inverse_scale[scale_group]);
    int const output_column =
        final_group * kPairsPerBlock +
        thread * 2;
    *reinterpret_cast<uint16_t*>(
        output +
        static_cast<int64_t>(final_route) *
            kQwenIntermediate +
        output_column) = pair;
  }
#else
  CUTE_INVALID_CONTROL_PATH(
      "Qwen3.5 split-K W1 requires the sm_120a target");
#endif
}

template <bool UsePdl, bool Wide, bool Indirect, class TmaLoadA>
__global__ __launch_bounds__(
    Wide ? kGroupedFusedThreads : kGroupedFusedNarrowThreads, 2)
void grouped_w1_silu_mxfp8_kernel(
    uint8_t const* activation,
    uint8_t const* activation_scales,
    uint8_t const* weight,
    CUTLASS_GRID_CONSTANT TmaLoadA const tma_load_a,
    uint8_t const* weight_scales,
    int64_t const* expert_offsets,
    int64_t const* scale_offsets,
    uint8_t* output,
    uint8_t* output_scales,
    int num_experts,
    int32_t const* source_tokens,
    Qwen35GroupedMetadataDevice grouped_metadata) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  using FusedMainloop = std::conditional_t<
      Wide, GroupedFusedMainloop, GroupedFusedNarrowMainloop>;
  using FusedTiledMma = std::conditional_t<
      Wide, GroupedFusedTiledMma, GroupedFusedNarrowTiledMma>;
  using FusedMmaOp = std::conditional_t<
      Wide, GroupedFusedMmaOp, GroupedFusedNarrowMmaOp>;
  using FusedSmemCopyAtomA = std::conditional_t<
      Wide, GroupedFusedSmemCopyAtomA,
      GroupedFusedNarrowSmemCopyAtomA>;
  using FusedSmemCopyAtomB = std::conditional_t<
      Wide, GroupedFusedSmemCopyAtomB,
      GroupedFusedNarrowSmemCopyAtomB>;
  using FusedSmemLayoutA = std::conditional_t<
      Wide, GroupedFusedSmemLayoutA,
      GroupedFusedNarrowSmemLayoutA>;
  using FusedSmemLayoutB = std::conditional_t<
      Wide, GroupedFusedSmemLayoutB,
      GroupedFusedNarrowSmemLayoutB>;
  using FusedSharedStorage = std::conditional_t<
      Wide, GroupedFusedSharedStorage,
      GroupedFusedNarrowSharedStorage>;
  constexpr int kFusedTileM =
      Wide ? kGroupedFusedTileM : kGroupedFusedNarrowTileM;
  constexpr int kFusedThreads =
      Wide ? kGroupedFusedThreads : kGroupedFusedNarrowThreads;
  constexpr int kFusedPairsPerBlock =
      Wide ? kGroupedFusedPairsPerBlock
           : kGroupedFusedNarrowPairsPerBlock;
  constexpr int kFusedQuantPairs = kFusedPairsPerBlock / 2;
  if constexpr (UsePdl) {
    cutlass::arch::wait_on_dependent_grids();
  }
  constexpr int kHidden = 2048;
  constexpr int kIntermediate = 256;
  constexpr int kGateUp = 2 * kIntermediate;
  constexpr int kBlocks = kHidden / kMxScaleVectorSize;
  constexpr int kActivationBlocks = kIntermediate / kMxScaleVectorSize;
  constexpr uint32_t kTmaTransactionBytes =
      kFusedTileM * kGroupedFusedTileK * 3 / 4;

  int const expert = static_cast<int>(blockIdx.y);
  int64_t const row_start = expert_offsets[expert];
  int const rows = static_cast<int>(
      expert_offsets[expert + 1] - row_start);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    grouped_metadata.write(expert);
  }
  if (expert >= num_experts || rows <= 0) {
    if constexpr (UsePdl) {
      cutlass::arch::launch_dependent_grids();
    }
    return;
  }

  int const pair_block =
      static_cast<int>(blockIdx.x) * kFusedPairsPerBlock;
  int const thread = static_cast<int>(threadIdx.x);
  int const quant_pair = thread & (kFusedQuantPairs - 1);
  int const quant_row = thread / kFusedQuantPairs;
  (void)weight;
  uint8_t const* expert_weight_scales =
      weight_scales + static_cast<int64_t>(expert) *
          kGateUp * kBlocks;
  extern __shared__ char dynamic_shared[];
  auto& shared = *reinterpret_cast<FusedSharedStorage*>(
      dynamic_shared);
  auto s_a = cute::make_tensor(
      cute::make_smem_ptr(shared.a), FusedSmemLayoutA{});
  auto s_b = cute::make_tensor(
      cute::make_smem_ptr(shared.b), FusedSmemLayoutB{});
  using X = cute::Underscore;
  auto m_a = tma_load_a.get_tma_tensor(cute::make_shape(
      cute::make_shape(
          cute::Int<32>{}, cute::Int<2>{},
          cute::Int<8>{}, num_experts),
      cute::Int<kHidden>{}, cute::Int<1>{}));
  auto g_a_mkl = cute::local_tile(
      m_a,
      typename FusedMainloop::TileShape{},
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<cute::_1, X, cute::_1>{});
  int const global_m_tile = expert * (kGateUp / kFusedTileM) +
      static_cast<int>(blockIdx.x);
  auto g_a = g_a_mkl(
      cute::_, cute::_, global_m_tile, cute::_, 0);
  auto block_tma_a = tma_load_a.get_slice(0);
  auto tma_source_a = block_tma_a.partition_S(g_a);
  auto tma_destination_a = block_tma_a.partition_D(s_a);
  FusedTiledMma tiled_mma;
  FusedMainloop reference_mainloop;
  auto thread_mma = tiled_mma.get_thread_slice(thread);
  auto fragment_a = thread_mma.partition_fragment_A(s_a);
  auto fragment_b = thread_mma.partition_fragment_B(s_b);
  auto tiled_copy_a = cute::make_tiled_copy_A(
      FusedSmemCopyAtomA{}, tiled_mma);
  auto tiled_copy_b = cute::make_tiled_copy_B(
      FusedSmemCopyAtomB{}, tiled_mma);
  auto thread_copy_a = tiled_copy_a.get_thread_slice(thread);
  auto thread_copy_b = tiled_copy_b.get_thread_slice(thread);
  auto source_a = thread_copy_a.partition_S(
      cute::as_position_independent_swizzle_tensor(s_a));
  auto source_b = thread_copy_b.partition_S(
      cute::as_position_independent_swizzle_tensor(s_b));
  auto register_a = thread_copy_a.retile_D(fragment_a);
  auto register_b = thread_copy_b.retile_D(fragment_b);
  auto const sfa_layout =
      reference_mainloop.get_layoutSFA_TV(tiled_mma);
  auto const sfb_layout =
      reference_mainloop.get_layoutSFB_TV(tiled_mma);
  int const local_sfa_row =
      sfa_layout(thread, 0) % kFusedTileM;
  int const sfb_row = sfb_layout(thread, 0) % kTileN;
  auto c_identity = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kFusedTileM>{}, cute::_8{}));
  auto c_coordinates = cute::composition(
      c_identity, tiled_mma.get_layoutC_TV());

  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  int tma_phase = 0;
  if (thread == 0) {
    TmaBarrier::init(&shared.tma_barrier, 1);
    cute::prefetch_tma_descriptor(
        tma_load_a.get_tma_descriptor());
  }
  __syncthreads();

  constexpr int kRowTilesPerCta = 1;
  for (int row_base = 0;
       row_base < rows;
       row_base += kTileN * kRowTilesPerCta) {
    float accum0[kRowTilesPerCta]{};
    float accum1[kRowTilesPerCta]{};
    float accum2[kRowTilesPerCta]{};
    float accum3[kRowTilesPerCta]{};
    int const row_tiles = min(
        kRowTilesPerCta,
        (rows - row_base + kTileN - 1) / kTileN);
    int weight_row;
    if constexpr (Wide) {
      int const physical_chunk = local_sfa_row / 32;
      int const logical_pair =
          pair_block + physical_chunk / 2 * 32 +
          local_sfa_row % 32;
      weight_row = logical_pair +
          (physical_chunk & 1) * kIntermediate;
    } else {
      weight_row = pair_block +
          (local_sfa_row < kFusedPairsPerBlock
               ? local_sfa_row
               : kIntermediate + local_sfa_row -
                     kFusedPairsPerBlock);
    }
    if (thread < kRowTilesPerCta * kTileN) {
      int const row_tile = thread / kTileN;
      int const local_row = thread % kTileN;
      int const routed_row =
          row_base + row_tile * kTileN + local_row;
      if (routed_row < rows) {
        if constexpr (Indirect) {
          shared.source_rows[row_tile][local_row] =
              source_tokens[row_start + routed_row];
        } else {
          shared.source_rows[row_tile][local_row] =
              static_cast<int32_t>(row_start + routed_row);
        }
      } else {
        shared.source_rows[row_tile][local_row] = 0;
      }
    }
    __syncthreads();

    for (int k_tile = 0;
         k_tile < kHidden;
         k_tile += kGroupedFusedTileK) {
      if (thread == 0) {
        TmaBarrier::arrive_and_expect_tx(
            &shared.tma_barrier, kTmaTransactionBytes);
        cute::copy(
            tma_load_a.with(shared.tma_barrier),
            tma_source_a(
                cute::_, cute::_, cute::_,
                k_tile / kGroupedFusedTileK),
            tma_destination_a);
      }
      CUTE_UNROLL
      for (int row_tile = 0;
           row_tile < kRowTilesPerCta;
           ++row_tile) {
        if (row_tile >= row_tiles) {
          continue;
        }
        constexpr int kActivationSegments =
            kGroupedFusedTileK / sizeof(uint4);
        for (int segment = thread;
             segment < kTileN * kActivationSegments;
             segment += kFusedThreads) {
          int const local_row = segment / kActivationSegments;
          int const segment_in_row = segment % kActivationSegments;
          int const column = segment_in_row * sizeof(uint4);
          int const source_column =
              (segment_in_row ^ (local_row & 7)) * sizeof(uint4);
          int const source_row =
              row_base + row_tile * kTileN + local_row;
          int const activation_row =
              shared.source_rows[row_tile][local_row];
          int const destination = s_b.layout()(local_row, column);
          if (source_row < rows) {
            copy_global_to_shared_16<16>(
                shared.b + destination,
                activation +
                    static_cast<int64_t>(activation_row) * kHidden +
                    k_tile + source_column);
          } else {
            *reinterpret_cast<uint4*>(shared.b + destination) = uint4{};
          }
        }
        commit_global_to_shared();
        int const first_global_k_block =
            k_tile / kMxScaleVectorSize;
        uint32_t const sfa_codes0 =
            *reinterpret_cast<uint32_t const*>(
                expert_weight_scales + scale_offset(
                    weight_row, first_global_k_block, kBlocks));
        uint32_t const sfa_codes1 =
            *reinterpret_cast<uint32_t const*>(
                expert_weight_scales + scale_offset(
                    weight_row,
                    first_global_k_block + 4,
                    kBlocks));
        int const local_scale_row =
            row_base + row_tile * kTileN + sfb_row;
        uint32_t sfb_codes0 = 0x7f7f7f7fu;
        uint32_t sfb_codes1 = 0x7f7f7f7fu;
        if (local_scale_row < rows) {
          int64_t activation_scale_row;
          if constexpr (Indirect) {
            activation_scale_row =
                shared.source_rows[row_tile][sfb_row];
          } else {
            activation_scale_row =
                scale_offsets[expert] + local_scale_row;
          }
          sfb_codes0 = *reinterpret_cast<uint32_t const*>(
              activation_scales + scale_offset(
                  activation_scale_row,
                  first_global_k_block,
                  kBlocks));
          sfb_codes1 = *reinterpret_cast<uint32_t const*>(
              activation_scales + scale_offset(
                  activation_scale_row,
                  first_global_k_block + 4,
                  kBlocks));
        }
        wait_global_to_shared();
        TmaBarrier::wait(&shared.tma_barrier, tma_phase);
        tma_phase ^= 1;
        __syncthreads();

        CUTE_UNROLL
        for (int k_block = 0;
             k_block < kGroupedFusedTileK / kMxScaleVectorSize;
             ++k_block) {
          cute::copy(
              tiled_copy_a,
              source_a(_, _, k_block),
              register_a(_, _, k_block));
          cute::copy(
              tiled_copy_b,
              source_b(_, _, k_block),
              register_b(_, _, k_block));
          auto a_words = cute::recast<uint32_t>(
              fragment_a(_, _, k_block));
          auto b_words = cute::recast<uint32_t>(
              fragment_b(_, _, k_block));
          uint32_t const sfa_codes =
              k_block < 4 ? sfa_codes0 : sfa_codes1;
          uint32_t const sfb_codes =
              k_block < 4 ? sfb_codes0 : sfb_codes1;
          int const scale_shift = (k_block & 3) * 8;
          uint8_t const sfa = static_cast<uint8_t>(
              sfa_codes >> scale_shift);
          uint8_t const sfb = static_cast<uint8_t>(
              sfb_codes >> scale_shift);
          FusedMmaOp::fma(
              accum0[row_tile], accum1[row_tile],
              accum2[row_tile], accum3[row_tile],
              a_words(0), a_words(1), a_words(2), a_words(3),
              b_words(0), b_words(1),
              accum0[row_tile], accum1[row_tile],
              accum2[row_tile], accum3[row_tile],
              sfa, sfb);
        }
        __syncthreads();
      }
    }

    CUTE_UNROLL
    for (int row_tile = 0;
         row_tile < kRowTilesPerCta;
         ++row_tile) {
      if (row_tile >= row_tiles) {
        continue;
      }
      float const accumulator_values[4] = {
          accum0[row_tile], accum1[row_tile],
          accum2[row_tile], accum3[row_tile]};
      CUTE_UNROLL
      for (int value = 0; value < 4; ++value) {
        auto const coordinate = c_coordinates(thread, value);
        int const column =
            static_cast<int>(cute::get<0>(coordinate));
        int const row =
            static_cast<int>(cute::get<1>(coordinate));
        shared.accumulators[row][column] = accumulator_values[value];
      }
      __syncthreads();

      int const output_row =
          row_base + row_tile * kTileN + quant_row;
      int const output_column = quant_pair * 2;
      int const gate_column = Wide
          ? output_column / 32 * 64 + output_column % 32
          : output_column;
      int const up_column = gate_column +
          (Wide ? 32 : kFusedPairsPerBlock);
      float value0 = 0.0f;
      float value1 = 0.0f;
      if (output_row < rows) {
        float gate0 = round_bfloat16(
            shared.accumulators[quant_row][gate_column]);
        float gate1 = round_bfloat16(
            shared.accumulators[quant_row][gate_column + 1]);
        float const up0 = round_bfloat16(
            shared.accumulators[quant_row][up_column]);
        float const up1 = round_bfloat16(
            shared.accumulators[quant_row][up_column + 1]);
        gate0 = round_bfloat16(gate0 / (1.0f + expf(-gate0)));
        gate1 = round_bfloat16(gate1 / (1.0f + expf(-gate1)));
        value0 = round_bfloat16(gate0 * up0);
        value1 = round_bfloat16(gate1 * up1);
      }
      float absmax = fmaxf(fabsf(value0), fabsf(value1));
      for (int offset = 8; offset > 0; offset >>= 1) {
        absmax = fmaxf(
            absmax,
            __shfl_xor_sync(0xffffffffu, absmax, offset, 16));
      }
      float inverse_scale = 1.0f;
      int const quant_scale_group = quant_pair / 16;
      if ((quant_pair & 15) == 0) {
        float const raw_scale = fmaxf(absmax / 448.0f, 1.0e-30f);
        uint32_t scale_bits = __float_as_uint(raw_scale);
        scale_bits = (scale_bits + 0x007fffffu) & 0x7f800000u;
        inverse_scale = 1.0f / __uint_as_float(scale_bits);
        if (output_row < rows) {
          output_scales[scale_offset(
              scale_offsets[expert] + output_row,
              pair_block / kMxScaleVectorSize + quant_scale_group,
              kActivationBlocks)] =
                  static_cast<uint8_t>(scale_bits >> 23);
        }
      }
      inverse_scale = __shfl_sync(
          0xffffffffu, inverse_scale, 0, 16);
      if (output_row < rows) {
        uint16_t const pair = quantize_e4m3_pair(
            value0 * inverse_scale,
            value1 * inverse_scale);
        *reinterpret_cast<uint16_t*>(
            output +
            (row_start + output_row) * kIntermediate +
            pair_block + output_column) = pair;
      }
    }
  }
  if constexpr (UsePdl) {
    cutlass::arch::launch_dependent_grids();
  }
#else
  CUTE_INVALID_CONTROL_PATH(
      "fused grouped W1 requires the sm_120a target");
#endif
}

template <class Id>
__global__ __launch_bounds__(kFusedSwapThreads, 2)
void route_w6a8_silu_mxfp8_kernel(
    uint8_t const* activation,
    uint8_t const* logical_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    Id const* topk_ids,
    uint8_t* output,
    uint8_t* output_scales,
    int activation_rows,
    int routed_topk,
    int include_shared,
    int num_experts,
    int weight_row_bytes,
    int n,
    int k) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  int const route = blockIdx.y;
  int const routes_per_token = routed_topk + include_shared;
  int const token = route / routes_per_token;
  int const route_lane = route - token * routes_per_token;
  int const expert =
      include_shared && route_lane == routed_topk
      ? num_experts - 1
      : static_cast<int>(
            topk_ids[static_cast<int64_t>(token) * routed_topk +
                     route_lane]);
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  int const source_row =
      activation_rows == gridDim.y ? route : token;
  int const intermediate = n / 2;
  int const pair_block =
      blockIdx.x * kFusedSwapPairsPerBlock;
  int const thread = threadIdx.x;
  int const warp = thread / 32;
  int const lane = thread % 32;
  int const pair_base =
      pair_block + warp * kFusedSwapPairsPerWarp;
  int const k_blocks = k / kMxScaleVectorSize;
  int const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;

  uint8_t const* expert_weight =
      weight +
      static_cast<int64_t>(expert) * n *
          weight_row_bytes;
  uint8_t const* expert_scales =
      weight_scales +
      static_cast<int64_t>(expert) * n * packed_k_blocks;
  uint8_t const* source_activation =
      activation + static_cast<int64_t>(source_row) * k;
  uint8_t const* source_scales =
      logical_scales +
      static_cast<int64_t>(source_row) * k_blocks;

  extern __shared__ char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<FusedSwapSharedStorage*>(dynamic_shared);
  auto s_a = cute::make_tensor(
      cute::make_smem_ptr(shared.a), SwapSmemLayoutA{});
  auto s_b = cute::make_tensor(
      cute::make_smem_ptr(shared.b[warp]), SwapSmemLayoutB{});

  SwapTiledMma tiled_mma;
  SwapMainloop reference_mainloop;
  auto thread_mma = tiled_mma.get_thread_slice(lane);
  auto fragment_a = thread_mma.partition_fragment_A(s_a);
  auto fragment_b = thread_mma.partition_fragment_B(s_b);
  auto tiled_copy_a =
      cute::make_tiled_copy_A(SwapSmemCopyAtomA{}, tiled_mma);
  auto tiled_copy_b =
      cute::make_tiled_copy_B(SwapSmemCopyAtomB{}, tiled_mma);
  auto thread_copy_a = tiled_copy_a.get_thread_slice(lane);
  auto thread_copy_b = tiled_copy_b.get_thread_slice(lane);
  auto source_a = thread_copy_a.partition_S(
      cute::as_position_independent_swizzle_tensor(s_a));
  auto source_b = thread_copy_b.partition_S(
      cute::as_position_independent_swizzle_tensor(s_b));
  auto register_a = thread_copy_a.retile_D(fragment_a);
  auto register_b = thread_copy_b.retile_D(fragment_b);

  constexpr int kActivationSegments = kTileK / sizeof(uint4);
  for (int segment = thread;
       segment < kSwapTileM * kActivationSegments;
       segment += kFusedSwapThreads) {
    int const local_row = segment / kActivationSegments;
    int const column = segment % kActivationSegments * sizeof(uint4);
    int const destination = s_a.layout()(local_row, column);
    *reinterpret_cast<uint4*>(shared.a + destination) = uint4{};
  }
  __syncthreads();

  auto const sfb_layout =
      reference_mainloop.get_layoutSFB_TV(tiled_mma);
  int const sfb_row = sfb_layout(lane, 0) % kSwapWarpN;
  float accum0 = 0.0f;
  float accum1 = 0.0f;
  float accum2 = 0.0f;
  float accum3 = 0.0f;

  for (int k_tile = 0; k_tile < k; k_tile += kTileK) {
    for (int segment = thread;
         segment < kActivationSegments;
         segment += kFusedSwapThreads) {
      int const column = segment * sizeof(uint4);
      int const destination = s_a.layout()(0, column);
      copy_global_to_shared_16<16>(
          shared.a + destination,
          source_activation + k_tile + column);
    }

    constexpr int segments_per_row =
        kTileK / kFp6ValuesPerSegment;
    for (int segment = thread;
         segment <
             kFusedSwapWarps * kSwapWarpN * segments_per_row;
         segment += kFusedSwapThreads) {
      constexpr int segments_per_warp =
          kSwapWarpN * segments_per_row;
      int const destination_warp = segment / segments_per_warp;
      int const local_segment = segment % segments_per_warp;
      int const row = local_segment / segments_per_row;
      int const segment_in_row =
          local_segment % segments_per_row;
      int const weight_row =
          pair_block +
          destination_warp * kFusedSwapPairsPerWarp +
          (row < kFusedSwapPairsPerWarp
               ? row
               : intermediate + row - kFusedSwapPairsPerWarp);
      int const source_segment =
          segment_in_row ^ (row & 7);
      uint8_t const* source =
          expert_weight +
          static_cast<int64_t>(weight_row) * weight_row_bytes +
          (weight_row_bytes == k
               ? k_tile +
                     source_segment *
                         kFp6SharedBytesPerSegment
               : static_cast<int64_t>(k_tile) * 3 / 4 +
                     source_segment * kFp6BytesPerSegment);
      int const destination_offset = s_b.layout()(
          row, segment_in_row * kFp6SharedBytesPerSegment);
      if (weight_row_bytes == k) {
        copy_global_to_shared_16<16>(
            shared.b[destination_warp] + destination_offset,
            source);
      } else {
        uint8_t* destination =
            shared.b[destination_warp] + destination_offset;
        copy_fp6_segment_to_shared(
            destination, source);
      }
    }
    commit_global_to_shared();
    wait_global_to_shared();
    __syncthreads();

    CUTE_UNROLL
    for (int k_block = 0; k_block < kTileK / 32; ++k_block) {
      cute::copy(
          tiled_copy_a,
          source_a(_, _, k_block),
          register_a(_, _, k_block));
      cute::copy(
          tiled_copy_b,
          source_b(_, _, k_block),
          register_b(_, _, k_block));

      auto a_words =
          cute::recast<uint32_t>(fragment_a(_, _, k_block));
      auto b_words =
          cute::recast<uint32_t>(fragment_b(_, _, k_block));
      int const global_k_block = k_tile / 32 + k_block;
      int const weight_row =
          pair_base +
          (sfb_row < kFusedSwapPairsPerWarp
               ? sfb_row
               : intermediate + sfb_row -
                     kFusedSwapPairsPerWarp);
      uint8_t const block_sfa = source_scales[global_k_block];
      uint8_t const sfb = expert_scales[scale_offset(
          weight_row, global_k_block, k_blocks)];
      SwapMmaOp::fma(
          accum0, accum1, accum2, accum3,
          a_words(0), a_words(1), a_words(2), a_words(3),
          b_words(0), b_words(1),
          accum0, accum1, accum2, accum3,
          block_sfa, sfb);
    }
    __syncthreads();
  }

  float const up0 = __shfl_sync(
      0xffffffffu, accum0, (lane & 1) + 2);
  float const up1 = __shfl_sync(
      0xffffffffu, accum1, (lane & 1) + 2);
  float value0 = 0.0f;
  float value1 = 0.0f;
  if (lane < 2) {
    float gate0 = round_bfloat16(accum0);
    float gate1 = round_bfloat16(accum1);
    float const rounded_up0 = round_bfloat16(up0);
    float const rounded_up1 = round_bfloat16(up1);
    gate0 = round_bfloat16(gate0 / (1.0f + expf(-gate0)));
    gate1 = round_bfloat16(gate1 / (1.0f + expf(-gate1)));
    value0 = round_bfloat16(gate0 * rounded_up0);
    value1 = round_bfloat16(gate1 * rounded_up1);
  }

  float absmax = lane < 2
      ? fmaxf(fabsf(value0), fabsf(value1))
      : 0.0f;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    absmax = fmaxf(
        absmax,
        __shfl_down_sync(0xffffffffu, absmax, offset));
  }
  if (lane == 0) {
    shared.warp_absmax[warp] = absmax;
  }
  __syncthreads();

  if (warp == 0) {
    float block_absmax =
        lane < kFusedSwapWarps ? shared.warp_absmax[lane] : 0.0f;
#pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
      block_absmax = fmaxf(
          block_absmax,
          __shfl_down_sync(
              0xffffffffu, block_absmax, offset));
    }
    if (lane == 0) {
      float const raw_scale =
          fmaxf(block_absmax / 448.0f, 1.0e-30f);
      uint32_t scale_bits = __float_as_uint(raw_scale);
      scale_bits =
          (scale_bits + 0x007fffffu) & 0x7f800000u;
      shared.scale_code =
          static_cast<uint8_t>(scale_bits >> 23);
      shared.inverse_scale =
          1.0f / __uint_as_float(scale_bits);
      output_scales[
          static_cast<int64_t>(route) * gridDim.x +
          blockIdx.x] = shared.scale_code;
    }
  }
  __syncthreads();

  if (lane < 2) {
    uint16_t const pair = quantize_e4m3_pair(
        value0 * shared.inverse_scale,
        value1 * shared.inverse_scale);
    int const output_column =
        pair_base + lane * 2;
    *reinterpret_cast<uint16_t*>(
        output +
        static_cast<int64_t>(route) * intermediate +
        output_column) = pair;
  }
#else
  CUTE_INVALID_CONTROL_PATH(
      "fused swapAB MXFP6 MoE requires the sm_120a target");
#endif
}

__device__ __forceinline__ float load_direct_gate(
    void const* gate,
    int gate_type,
    int64_t index) {
  if (gate_type == 1) {
    return static_cast<float const*>(gate)[index];
  }
  if (gate_type == 2) {
    return static_cast<float>(
        static_cast<cutlass::half_t const*>(gate)[index]);
  }
  return static_cast<float>(
      static_cast<cutlass::bfloat16_t const*>(gate)[index]);
}

template <bool SeparateShared = false>
__global__ __launch_bounds__(kReduceThreads, 1)
void qwen35_w2_splitk_reduce_kernel(
    uint8_t const* activation,
    uint8_t const* logical_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    uint8_t const* shared_weight,
    uint8_t const* shared_weight_scales,
    int32_t const* topk_ids,
    float const* topk_weights,
    cutlass::bfloat16_t const* shared_gate,
    float* partial,
    cutlass::bfloat16_t* output) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  cutlass::arch::wait_on_dependent_grids();
  constexpr int kQwenHidden = 2048;
  constexpr int kQwenK = 256;
  constexpr int kQwenRoutes = kReduceMaxRoutes;
  constexpr int kQwenTiles =
      kQwenHidden / kReduceTileM;
  constexpr int kSplits = 2;
  constexpr int kWorkItems = kQwenTiles * kSplits;
  static_assert(kWorkItems == 256);

  int const work = blockIdx.x;
  int const split = work % kSplits;
  int const output_tile = work / kSplits;
  int const tile_m = output_tile * kReduceTileM;
  int const thread = threadIdx.x;
  int const warp = thread / 32;
  int const lane = thread % 32;
  bool const shared_route = warp == kQwenRoutes - 1;
  int expert = !shared_route
      ? topk_ids[warp]
      : 256;
  constexpr int kNumExperts = SeparateShared ? 256 : 257;
  if (expert < 0 || expert >= kNumExperts) {
    expert = 0;
  }
  constexpr int kBlocks = kQwenK / kMxScaleVectorSize;
  uint8_t const* expert_scales =
      SeparateShared && shared_route
      ? shared_weight_scales
      : weight_scales +
            static_cast<int64_t>(expert) *
                kQwenHidden * kBlocks;
  uint8_t const* source_scales =
      logical_scales + warp * kBlocks;

  extern __shared__ char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<ReduceSharedStorage*>(
          dynamic_shared);
  auto s_a = cute::make_tensor(
      cute::make_smem_ptr(shared.a[warp]),
      ReduceSmemLayoutA{});
  auto s_b = cute::make_tensor(
      cute::make_smem_ptr(shared.b[warp]),
      ReduceSmemLayoutB{});
  ReduceTiledMma tiled_mma;
  ReduceMainloop reference_mainloop;
  auto thread_mma = tiled_mma.get_thread_slice(lane);
  auto fragment_a = thread_mma.partition_fragment_A(s_a);
  auto fragment_b = thread_mma.partition_fragment_B(s_b);
  auto tiled_copy_a = cute::make_tiled_copy_A(
      ReduceSmemCopyAtomA{}, tiled_mma);
  auto tiled_copy_b = cute::make_tiled_copy_B(
      ReduceSmemCopyAtomB{}, tiled_mma);
  auto thread_copy_a =
      tiled_copy_a.get_thread_slice(lane);
  auto thread_copy_b =
      tiled_copy_b.get_thread_slice(lane);
  auto source_a = thread_copy_a.partition_S(
      cute::as_position_independent_swizzle_tensor(s_a));
  auto source_b = thread_copy_b.partition_S(
      cute::as_position_independent_swizzle_tensor(s_b));
  auto register_a = thread_copy_a.retile_D(fragment_a);
  auto register_b = thread_copy_b.retile_D(fragment_b);
  auto const sfa_layout =
      reference_mainloop.get_layoutSFA_TV(tiled_mma);
  int const sfa_row =
      sfa_layout(lane, 0) % kReduceTileM;

  for (int vector = thread;
       vector <
           static_cast<int>(
               sizeof(shared.b) / sizeof(uint4));
       vector += kReduceThreads) {
    reinterpret_cast<uint4*>(shared.b)[vector] =
        uint4{0, 0, 0, 0};
  }
  __syncthreads();

  int const k_tile = split * kTileK;
  constexpr int segments_per_row =
      kTileK / kFp6ValuesPerSegment;
  constexpr int segments_per_warp =
      kReduceTileM * segments_per_row;
  for (int segment = thread;
       segment < kQwenRoutes * segments_per_warp;
       segment += kReduceThreads) {
    int const destination_warp =
        segment / segments_per_warp;
    int const local_segment =
        segment % segments_per_warp;
    int const row =
        local_segment / segments_per_row;
    int const segment_in_row =
        local_segment % segments_per_row;
    bool const destination_shared =
        destination_warp == kQwenRoutes - 1;
    int destination_expert =
        !destination_shared
        ? topk_ids[destination_warp]
        : 256;
    if (destination_expert < 0 ||
        destination_expert >= kNumExperts) {
      destination_expert = 0;
    }
    uint8_t const* destination_weight =
        SeparateShared && destination_shared
        ? shared_weight
        : weight +
              static_cast<int64_t>(destination_expert) *
                  kQwenHidden * kQwenK * 3 / 4;
    uint8_t const* source =
        destination_weight +
        static_cast<int64_t>(tile_m + row) *
            kQwenK * 3 / 4 +
        static_cast<int64_t>(k_tile) * 3 / 4 +
        segment_in_row * kFp6BytesPerSegment;
    auto const* source_words =
        reinterpret_cast<uint32_t const*>(source);
    uint4 const packed{
        source_words[0],
        source_words[1],
        source_words[2],
        0};
    ReduceSmemLayoutA layout_a;
    int const destination_offset = layout_a(
        row,
        segment_in_row * kFp6SharedBytesPerSegment);
    *reinterpret_cast<uint4*>(
        shared.a[destination_warp] +
        destination_offset) = packed;
  }

  constexpr int activation_vectors =
      kTileK / sizeof(uint4);
  for (int vector = thread;
       vector <
           kQwenRoutes * activation_vectors;
       vector += kReduceThreads) {
    int const destination_warp =
        vector / activation_vectors;
    int const vector_in_row =
        vector % activation_vectors;
    int const column =
        vector_in_row * sizeof(uint4);
    ReduceSmemLayoutB layout_b;
    int const destination_offset =
        layout_b(0, column);
    uint8_t const* source =
        activation +
        destination_warp * kQwenK +
        k_tile + column;
    *reinterpret_cast<uint4*>(
        shared.b[destination_warp] +
        destination_offset) =
        *reinterpret_cast<uint4 const*>(source);
  }
  int const first_global_k_block =
      k_tile / kMxScaleVectorSize;
  uint32_t const sfa_codes =
      *reinterpret_cast<uint32_t const*>(
          expert_scales + scale_offset(
              tile_m + sfa_row,
              first_global_k_block,
              kBlocks));
  uint32_t const sfb_codes =
      *reinterpret_cast<uint32_t const*>(
          source_scales + first_global_k_block);
  __syncthreads();

  float accum0 = 0.0f;
  float accum1 = 0.0f;
  float accum2 = 0.0f;
  float accum3 = 0.0f;
  CUTE_UNROLL
  for (int k_block = 0;
       k_block < kTileK / kMxScaleVectorSize;
       ++k_block) {
    cute::copy(
        tiled_copy_a,
        source_a(_, _, k_block),
        register_a(_, _, k_block));
    cute::copy(
        tiled_copy_b,
        source_b(_, _, k_block),
        register_b(_, _, k_block));
    auto a_words =
        cute::recast<uint32_t>(
            fragment_a(_, _, k_block));
    auto b_words =
        cute::recast<uint32_t>(
            fragment_b(_, _, k_block));
    int const scale_shift = k_block * 8;
    uint8_t const sfa = static_cast<uint8_t>(
        sfa_codes >> scale_shift);
    uint8_t const sfb = static_cast<uint8_t>(
        sfb_codes >> scale_shift);
    ReduceMmaOp::fma(
        accum0,
        accum1,
        accum2,
        accum3,
        a_words(0),
        a_words(1),
        a_words(2),
        a_words(3),
        b_words(0),
        b_words(1),
        accum0,
        accum1,
        accum2,
        accum3,
        sfa,
        sfb);
  }

  if ((lane & 3) == 0) {
    int const row0 = lane / 4;
    int const row1 = row0 + 8;
    int64_t const partial_base =
        (static_cast<int64_t>(work) *
             kQwenRoutes +
         warp) *
        kReduceTileM;
    partial[partial_base + row0] = accum0;
    partial[partial_base + row1] = accum2;
  }

  cooperative_groups::this_grid().sync();

  if (work >= kQwenTiles ||
      warp != 0 ||
      lane >= kReduceTileM) {
    return;
  }
  int const first_work = work * kSplits;
  float const shared_value = round_bfloat16(
      partial[
          (static_cast<int64_t>(first_work) *
               kQwenRoutes +
           (kQwenRoutes - 1)) *
              kReduceTileM +
          lane] +
      partial[
          (static_cast<int64_t>(first_work + 1) *
               kQwenRoutes +
           (kQwenRoutes - 1)) *
              kReduceTileM +
          lane]);
  float const shared_scale =
      1.0f / (1.0f + expf(
          -static_cast<float>(shared_gate[0])));
  float result = shared_scale * shared_value;
#pragma unroll
  for (int route = 0;
       route < kQwenRoutes - 1;
       ++route) {
    float const route_value = round_bfloat16(
        partial[
            (static_cast<int64_t>(first_work) *
                 kQwenRoutes +
             route) *
                kReduceTileM +
            lane] +
        partial[
            (static_cast<int64_t>(first_work + 1) *
                 kQwenRoutes +
             route) *
                kReduceTileM +
            lane]);
    result += topk_weights[route] * route_value;
  }
  output[work * kReduceTileM + lane] =
      static_cast<cutlass::bfloat16_t>(result);
#else
  CUTE_INVALID_CONTROL_PATH(
      "Qwen3.5 split-K W2 requires the sm_120a target");
#endif
}

template <
    class Id,
    class ElementD,
    bool UsePdl = false,
    bool ExternalShared = false,
    bool SeparateShared = false,
    bool VectorizedPackedLoads = false>
__global__ __launch_bounds__(kReduceThreads, 1)
void route_w6a8_reduce_kernel(
    uint8_t const* activation,
    uint8_t const* logical_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    uint8_t const* shared_weight,
    uint8_t const* shared_weight_scales,
    Id const* topk_ids,
    float const* topk_weights,
    void const* shared_gate,
    ElementD const* external_shared_output,
    ElementD* output,
    int gate_type,
    int gate_stride,
    int gate_column,
    int num_experts,
    int weight_row_bytes,
    int n,
    int k) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  static_assert(!(ExternalShared && SeparateShared));
  if constexpr (UsePdl) {
    cutlass::arch::wait_on_dependent_grids();
  }
  int const token = blockIdx.y;
  int const tile_m = blockIdx.x * kReduceTileM;
  int const thread = threadIdx.x;
  int const warp = thread / 32;
  int const lane = thread % 32;
  constexpr int routes_per_token =
      ExternalShared ? kReduceMaxRoutes - 1 : kReduceMaxRoutes;
  constexpr int kernel_threads = routes_per_token * 32;
  int const route = token * routes_per_token + warp;
  bool const shared_route =
      !ExternalShared && warp == kReduceMaxRoutes - 1;
  int expert = shared_route && SeparateShared ? 0 : num_experts - 1;
  if constexpr (ExternalShared) {
    expert = static_cast<int>(
        topk_ids[
            static_cast<int64_t>(token) *
                (kReduceMaxRoutes - 1) +
            warp]);
  } else if (warp < kReduceMaxRoutes - 1) {
    expert = static_cast<int>(
        topk_ids[
            static_cast<int64_t>(token) *
                (kReduceMaxRoutes - 1) +
            warp]);
  }
  bool const valid_expert =
      (SeparateShared && shared_route) ||
      (expert >= 0 && expert < num_experts);
  if (!valid_expert) {
    expert = 0;
  }

  int const k_blocks = k / kMxScaleVectorSize;
  int const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  uint8_t const* expert_scales =
      SeparateShared && shared_route
      ? shared_weight_scales
      : weight_scales +
            static_cast<int64_t>(expert) * n * packed_k_blocks;
  uint8_t const* source_scales =
      logical_scales +
      static_cast<int64_t>(route) * k_blocks;

  extern __shared__ char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<ReduceSharedStorage*>(dynamic_shared);
  auto s_a = cute::make_tensor(
      cute::make_smem_ptr(shared.a[warp]),
      ReduceSmemLayoutA{});
  auto s_b = cute::make_tensor(
      cute::make_smem_ptr(shared.b[warp]),
      ReduceSmemLayoutB{});

  ReduceTiledMma tiled_mma;
  ReduceMainloop reference_mainloop;
  auto thread_mma = tiled_mma.get_thread_slice(lane);
  auto fragment_a = thread_mma.partition_fragment_A(s_a);
  auto fragment_b = thread_mma.partition_fragment_B(s_b);
  auto tiled_copy_a =
      cute::make_tiled_copy_A(ReduceSmemCopyAtomA{}, tiled_mma);
  auto tiled_copy_b =
      cute::make_tiled_copy_B(ReduceSmemCopyAtomB{}, tiled_mma);
  auto thread_copy_a = tiled_copy_a.get_thread_slice(lane);
  auto thread_copy_b = tiled_copy_b.get_thread_slice(lane);
  auto source_a = thread_copy_a.partition_S(
      cute::as_position_independent_swizzle_tensor(s_a));
  auto source_b = thread_copy_b.partition_S(
      cute::as_position_independent_swizzle_tensor(s_b));
  auto register_a = thread_copy_a.retile_D(fragment_a);
  auto register_b = thread_copy_b.retile_D(fragment_b);

  for (int vector = thread;
       vector < routes_per_token *
           static_cast<int>(sizeof(shared.b[0]) / sizeof(uint4));
       vector += kernel_threads) {
    reinterpret_cast<uint4*>(shared.b)[vector] =
        uint4{0, 0, 0, 0};
  }
  __syncthreads();

  auto const sfa_layout =
      reference_mainloop.get_layoutSFA_TV(tiled_mma);
  int const sfa_row = sfa_layout(lane, 0) % kReduceTileM;
  float accum0 = 0.0f;
  float accum1 = 0.0f;
  float accum2 = 0.0f;
  float accum3 = 0.0f;

  for (int k_tile = 0; k_tile < k; k_tile += kTileK) {
    constexpr int segments_per_row =
        kTileK / kFp6ValuesPerSegment;
    constexpr int segments_per_warp =
        kReduceTileM * segments_per_row;
    if constexpr (VectorizedPackedLoads) {
      static_assert(!ExternalShared);
      constexpr int packed_bytes_per_half_row =
          kTileK * 3 / 8;
      static_assert(packed_bytes_per_half_row == 3 * sizeof(uint4));
      int const row = lane / 2;
      int const half_row = lane % 2;
      uint8_t const* destination_weight =
          SeparateShared && shared_route
          ? shared_weight
          : weight +
                static_cast<int64_t>(expert) * n * weight_row_bytes;
      uint8_t const* source =
          destination_weight +
          static_cast<int64_t>(tile_m + row) * weight_row_bytes +
          static_cast<int64_t>(k_tile) * 3 / 4 +
          half_row * packed_bytes_per_half_row;
      ReduceSmemLayoutA layout_a;
      int const segment_base = half_row * 4;
      uint4 const packed0 =
          *reinterpret_cast<uint4 const*>(source);
      *reinterpret_cast<uint4*>(
          shared.a[warp] + layout_a(
              row, segment_base * kFp6SharedBytesPerSegment)) =
                  uint4{packed0.x, packed0.y, packed0.z, 0};
      uint4 const packed1 =
          *reinterpret_cast<uint4 const*>(source + sizeof(uint4));
      *reinterpret_cast<uint4*>(
          shared.a[warp] + layout_a(
              row, (segment_base + 1) * kFp6SharedBytesPerSegment)) =
                  uint4{packed0.w, packed1.x, packed1.y, 0};
      uint4 const packed2 =
          *reinterpret_cast<uint4 const*>(source + 2 * sizeof(uint4));
      *reinterpret_cast<uint4*>(
          shared.a[warp] + layout_a(
              row, (segment_base + 2) * kFp6SharedBytesPerSegment)) =
                  uint4{packed1.z, packed1.w, packed2.x, 0};
      *reinterpret_cast<uint4*>(
          shared.a[warp] + layout_a(
              row, (segment_base + 3) * kFp6SharedBytesPerSegment)) =
                  uint4{packed2.y, packed2.z, packed2.w, 0};
    } else {
      for (int segment = thread;
           segment < routes_per_token * segments_per_warp;
           segment += kernel_threads) {
        int const destination_warp = segment / segments_per_warp;
        int const local_segment = segment % segments_per_warp;
        int const row = local_segment / segments_per_row;
        int const segment_in_row = local_segment % segments_per_row;
        bool const destination_shared =
            !ExternalShared &&
            destination_warp == kReduceMaxRoutes - 1;
        int destination_expert =
            destination_shared && SeparateShared ? 0 : num_experts - 1;
        if constexpr (ExternalShared) {
          destination_expert = static_cast<int>(
              topk_ids[
                  static_cast<int64_t>(token) *
                      (kReduceMaxRoutes - 1) +
                  destination_warp]);
        } else if (destination_warp < kReduceMaxRoutes - 1) {
          destination_expert = static_cast<int>(
              topk_ids[
                  static_cast<int64_t>(token) *
                      (kReduceMaxRoutes - 1) +
                  destination_warp]);
        }
        bool const destination_valid =
            (SeparateShared && destination_shared) ||
            (destination_expert >= 0 &&
             destination_expert < num_experts);
        if (!destination_valid) {
          destination_expert = 0;
        }
        uint8_t const* destination_weight =
            SeparateShared && destination_shared
            ? shared_weight
            : weight +
                  static_cast<int64_t>(destination_expert) *
                      n * weight_row_bytes;
        uint8_t const* source =
            destination_weight +
            static_cast<int64_t>(tile_m + row) * weight_row_bytes +
            (weight_row_bytes == k
                 ? k_tile +
                       segment_in_row * kFp6SharedBytesPerSegment
                 : static_cast<int64_t>(k_tile) * 3 / 4 +
                       segment_in_row * kFp6BytesPerSegment);
        ReduceSmemLayoutA layout_a;
        int const destination_offset = layout_a(
            row,
            segment_in_row * kFp6SharedBytesPerSegment);
        if (weight_row_bytes == k) {
          *reinterpret_cast<uint4*>(
              shared.a[destination_warp] + destination_offset) =
                  *reinterpret_cast<uint4 const*>(source);
        } else {
          auto const* source_words =
              reinterpret_cast<uint32_t const*>(source);
          *reinterpret_cast<uint4*>(
              shared.a[destination_warp] + destination_offset) =
                  uint4{
                      source_words[0], source_words[1], source_words[2], 0};
        }
      }
    }

    constexpr int activation_vectors =
        kTileK / sizeof(uint4);
    for (int vector = thread;
         vector <
             routes_per_token * activation_vectors;
         vector += kernel_threads) {
      int const destination_warp =
          vector / activation_vectors;
      int const vector_in_row =
          vector % activation_vectors;
      int const column =
          vector_in_row * sizeof(uint4);
      ReduceSmemLayoutB layout_b;
      int const destination_offset = layout_b(0, column);
      uint8_t const* source =
          activation +
          (static_cast<int64_t>(token) *
               routes_per_token +
           destination_warp) *
              k +
          k_tile + column;
      *reinterpret_cast<uint4*>(
          shared.b[destination_warp] +
          destination_offset) =
          *reinterpret_cast<uint4 const*>(source);
    }
    __syncthreads();

    CUTE_UNROLL
    for (int k_block = 0; k_block < kTileK / 32; ++k_block) {
      cute::copy(
          tiled_copy_a,
          source_a(_, _, k_block),
          register_a(_, _, k_block));
      cute::copy(
          tiled_copy_b,
          source_b(_, _, k_block),
          register_b(_, _, k_block));
      auto a_words =
          cute::recast<uint32_t>(fragment_a(_, _, k_block));
      auto b_words =
          cute::recast<uint32_t>(fragment_b(_, _, k_block));
      int const global_k_block = k_tile / 32 + k_block;
      uint8_t const sfa = expert_scales[scale_offset(
          tile_m + sfa_row,
          global_k_block,
          k_blocks)];
      uint8_t const sfb = source_scales[global_k_block];
      ReduceMmaOp::fma(
          accum0, accum1, accum2, accum3,
          a_words(0), a_words(1), a_words(2), a_words(3),
          b_words(0), b_words(1),
          accum0, accum1, accum2, accum3,
          sfa, sfb);
    }
    __syncthreads();
  }

  float route_scale = 0.0f;
  if (lane == 0 && valid_expert) {
    if constexpr (ExternalShared) {
      route_scale = topk_weights[
          static_cast<int64_t>(token) *
              (kReduceMaxRoutes - 1) +
          warp];
    } else if (warp < kReduceMaxRoutes - 1) {
      route_scale = topk_weights[
          static_cast<int64_t>(token) *
              (kReduceMaxRoutes - 1) +
          warp];
    } else {
      float const gate = load_direct_gate(
          shared_gate,
          gate_type,
          static_cast<int64_t>(token) * gate_stride +
              gate_column);
      route_scale = 1.0f / (1.0f + expf(-gate));
    }
  }
  route_scale = __shfl_sync(
      0xffffffffu, route_scale, 0);
  if ((lane & 3) == 0) {
    int const local_m = lane / 4;
    shared.partial[warp][local_m] =
        route_scale *
        static_cast<float>(static_cast<ElementD>(accum0));
    shared.partial[warp][local_m + 8] =
        route_scale *
        static_cast<float>(static_cast<ElementD>(accum2));
  }
  __syncthreads();

  if (warp == 0 && lane < kReduceTileM) {
    float result;
    if constexpr (ExternalShared) {
      float const gate = load_direct_gate(
          shared_gate,
          gate_type,
          static_cast<int64_t>(token) * gate_stride +
              gate_column);
      result =
          (1.0f / (1.0f + expf(-gate))) *
          static_cast<float>(
              external_shared_output[
                  static_cast<int64_t>(token) * n +
                  tile_m + lane]);
    } else {
      result = shared.partial[kReduceMaxRoutes - 1][lane];
    }
#pragma unroll
    for (int routed = 0;
         routed < kReduceMaxRoutes - 1;
         ++routed) {
      result += shared.partial[routed][lane];
    }
    output[
        static_cast<int64_t>(token) * n +
        tile_m + lane] = static_cast<ElementD>(result);
  }
#else
  CUTE_INVALID_CONTROL_PATH(
      "fused W2/reduce MXFP6 MoE requires the sm_120a target");
#endif
}

template <class Id, class ElementD>
void launch(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    bool include_shared,
    cudaStream_t stream) {
  int const tokens = static_cast<int>(topk_ids.size(0));
  int const routed_topk = static_cast<int>(topk_ids.size(1));
  int const routes =
      tokens * (routed_topk + static_cast<int>(include_shared));
  int const n = static_cast<int>(weight.size(1));
  int const k = static_cast<int>(weight.size(2) * 4 / 3);
  if (routes <= kSwapRouteThreshold && k >= kSwapMinK) {
    dim3 const grid(n / kSwapTileN, routes, 1);
    route_w6a8_swapab_kernel<Id, ElementD>
        <<<grid, kSwapThreads, sizeof(SwapSharedStorage), stream>>>(
            static_cast<uint8_t const*>(activation.data_ptr()),
            logical_scales.data_ptr<uint8_t>(),
            weight.data_ptr<uint8_t>(),
            weight_scales.data_ptr<uint8_t>(),
            topk_ids.data_ptr<Id>(),
            reinterpret_cast<ElementD*>(output.data_ptr()),
            static_cast<int>(activation.size(0)),
            routed_topk,
            static_cast<int>(include_shared),
            static_cast<int>(weight.size(0)),
            n,
            k);
    return;
  }
  dim3 const grid(n / kTileM, routes, 1);
  route_w6a8_kernel<Id, ElementD>
      <<<grid, kThreads, sizeof(SharedStorage), stream>>>(
          static_cast<uint8_t const*>(activation.data_ptr()),
          logical_scales.data_ptr<uint8_t>(),
          weight.data_ptr<uint8_t>(),
          weight_scales.data_ptr<uint8_t>(),
          topk_ids.data_ptr<Id>(),
          reinterpret_cast<ElementD*>(output.data_ptr()),
          static_cast<int>(activation.size(0)),
          routed_topk,
          static_cast<int>(include_shared),
          static_cast<int>(weight.size(0)),
          n,
      k);
}

template <class ElementD>
void launch_grouped_static_impl(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& activation_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    bool use_pdl,
    cudaStream_t stream) {
  int const num_experts = static_cast<int>(weight.size(0));
  int const n = static_cast<int>(weight.size(1));
  int const k = static_cast<int>(weight.size(2) * 4 / 3);
  int const grid_m = grouped_w2_grid_m(n, output.size(0));
  dim3 const grid(grid_m, num_experts, 1);
  auto const* activation_ptr =
      static_cast<uint8_t const*>(activation.data_ptr());
  auto const* activation_scales_ptr =
      activation_scales.data_ptr<uint8_t>();
  auto const* weight_ptr = weight.data_ptr<uint8_t>();
  auto const* weight_scales_ptr =
      weight_scales.data_ptr<uint8_t>();
  auto const* expert_offsets_ptr =
      expert_offsets.data_ptr<int64_t>();
  auto const* scale_offsets_ptr =
      scale_offsets.data_ptr<int64_t>();
  auto* output_ptr =
      reinterpret_cast<ElementD*>(output.data_ptr());
  if (use_pdl) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id =
        cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        grouped_w6a8_static_kernel<ElementD, true>,
        activation_ptr,
        activation_scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        expert_offsets_ptr,
        scale_offsets_ptr,
        output_ptr,
        num_experts,
        n,
        k));
  } else {
    grouped_w6a8_static_kernel<ElementD, false>
        <<<grid,
           kThreads,
           sizeof(SharedStorage),
           stream>>>(
            activation_ptr,
            activation_scales_ptr,
            weight_ptr,
            weight_scales_ptr,
            expert_offsets_ptr,
            scale_offsets_ptr,
            output_ptr,
            num_experts,
            n,
            k);
  }
}

template <class ElementD>
void launch_grouped_static(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& activation_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    bool use_pdl,
    cudaStream_t stream) {
  launch_grouped_static_impl<ElementD>(
      output, activation, activation_scales, weight, weight_scales,
      expert_offsets, scale_offsets, use_pdl, stream);
}

template <class Id>
void launch_silu_mxfp8(
    at::Tensor& output,
    at::Tensor& output_scales,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    bool include_shared,
    cudaStream_t stream) {
  int const tokens = static_cast<int>(topk_ids.size(0));
  int const routed_topk = static_cast<int>(topk_ids.size(1));
  int const routes =
      tokens * (routed_topk + static_cast<int>(include_shared));
  int const n = static_cast<int>(weight.size(1));
  int const k = static_cast<int>(activation.size(1));
  dim3 const grid(
      n / 2 / kFusedSwapPairsPerBlock, routes, 1);
  route_w6a8_silu_mxfp8_kernel<Id>
      <<<grid,
         kFusedSwapThreads,
         sizeof(FusedSwapSharedStorage),
         stream>>>(
          static_cast<uint8_t const*>(activation.data_ptr()),
          logical_scales.data_ptr<uint8_t>(),
          weight.data_ptr<uint8_t>(),
          weight_scales.data_ptr<uint8_t>(),
          topk_ids.data_ptr<Id>(),
          output.data_ptr<uint8_t>(),
          output_scales.data_ptr<uint8_t>(),
          static_cast<int>(activation.size(0)),
          routed_topk,
          static_cast<int>(include_shared),
          static_cast<int>(weight.size(0)),
          static_cast<int>(weight.size(2)),
          n,
          k);
}

template <bool Wide, bool Indirect>
void launch_grouped_w1_silu_mxfp8_impl(
    at::Tensor& output,
    at::Tensor& output_scales,
    at::Tensor const& activation,
    at::Tensor const& activation_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    int32_t const* source_tokens,
    bool use_pdl,
    cudaStream_t stream,
    Qwen35GroupedMetadataDevice grouped_metadata = {}) {
  using FusedSharedStorage = std::conditional_t<
      Wide, GroupedFusedSharedStorage,
      GroupedFusedNarrowSharedStorage>;
  using FusedTmaInternalElementA =
      cutlass::detail::float_e3m2_unpacksmem_t;
  using FusedGmemTiledCopyA = cute::SM90_TMA_LOAD;
  using ReferenceFusedSmemLayoutA = std::conditional_t<
      Wide, ReferenceGroupedFusedSmemLayoutA,
      ReferenceGroupedFusedNarrowSmemLayoutA>;
  constexpr int kFusedThreads =
      Wide ? kGroupedFusedThreads : kGroupedFusedNarrowThreads;
  constexpr int kFusedPairsPerBlock =
      Wide ? kGroupedFusedPairsPerBlock
           : kGroupedFusedNarrowPairsPerBlock;
  constexpr int kFusedTileM =
      Wide ? kGroupedFusedTileM : kGroupedFusedNarrowTileM;
  constexpr int kHidden = 2048;
  constexpr int kGateUp = 512;
  dim3 const grid(
      256 / kFusedPairsPerBlock,
      static_cast<unsigned>(weight.size(0)),
      1);
  auto const* activation_ptr =
      static_cast<uint8_t const*>(activation.data_ptr());
  auto const* activation_scales_ptr =
      activation_scales.data_ptr<uint8_t>();
  auto const* weight_ptr = weight.data_ptr<uint8_t>();
  auto const* weight_scales_ptr = weight_scales.data_ptr<uint8_t>();
  auto const* expert_offsets_ptr = expert_offsets.data_ptr<int64_t>();
  auto const* scale_offsets_ptr = scale_offsets.data_ptr<int64_t>();
  auto* output_ptr = output.data_ptr<uint8_t>();
  auto* output_scales_ptr = output_scales.data_ptr<uint8_t>();
  int const num_experts = static_cast<int>(weight.size(0));
  auto const shape_a = cute::make_shape(
      cute::make_shape(
          cute::Int<32>{}, cute::Int<2>{},
          cute::Int<8>{}, num_experts),
      cute::Int<kHidden>{}, cute::Int<1>{});
  auto const stride_a = cute::make_stride(
      cute::make_stride(
          int64_t{kHidden},
          int64_t{256 * kHidden},
          int64_t{32 * kHidden},
          int64_t{kGateUp * kHidden}),
      cute::Int<1>{}, cute::Int<0>{});
  auto tensor_a = cute::make_tensor(
      reinterpret_cast<FusedTmaInternalElementA const*>(weight_ptr),
      cute::make_layout(shape_a, stride_a));
  auto tma_load_a = cute::make_tma_copy(
      FusedGmemTiledCopyA{},
      tensor_a,
      ReferenceFusedSmemLayoutA{}(
          cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(
          cute::Int<kFusedTileM>{},
          cute::Int<kGroupedFusedTileK>{}),
      cute::_1{});
  if (use_pdl) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(kFusedThreads);
    config.dynamicSmemBytes = sizeof(FusedSharedStorage);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        grouped_w1_silu_mxfp8_kernel<
            true, Wide, Indirect, decltype(tma_load_a)>,
        activation_ptr,
        activation_scales_ptr,
        weight_ptr,
        tma_load_a,
        weight_scales_ptr,
        expert_offsets_ptr,
        scale_offsets_ptr,
        output_ptr,
        output_scales_ptr,
        num_experts,
        source_tokens,
        grouped_metadata));
  } else {
    grouped_w1_silu_mxfp8_kernel<
        false, Wide, Indirect, decltype(tma_load_a)>
        <<<grid,
           kFusedThreads,
           sizeof(FusedSharedStorage),
           stream>>>(
            activation_ptr,
            activation_scales_ptr,
            weight_ptr,
            tma_load_a,
            weight_scales_ptr,
            expert_offsets_ptr,
            scale_offsets_ptr,
            output_ptr,
            output_scales_ptr,
            num_experts,
            source_tokens,
            grouped_metadata);
  }
}

template <bool Indirect>
void launch_grouped_w1_silu_mxfp8_shape(
    at::Tensor& output,
    at::Tensor& output_scales,
    at::Tensor const& activation,
    at::Tensor const& activation_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    int32_t const* source_tokens,
    bool use_pdl,
    cudaStream_t stream,
    Qwen35GroupedMetadataDevice grouped_metadata) {
  if (grouped_w1_wide(output.size(0))) {
    launch_grouped_w1_silu_mxfp8_impl<true, Indirect>(
        output, output_scales, activation, activation_scales,
        weight, weight_scales, expert_offsets, scale_offsets,
        source_tokens, use_pdl, stream, grouped_metadata);
  } else {
    launch_grouped_w1_silu_mxfp8_impl<false, Indirect>(
        output, output_scales, activation, activation_scales,
        weight, weight_scales, expert_offsets, scale_offsets,
        source_tokens, use_pdl, stream, grouped_metadata);
  }
}

void launch_grouped_w1_silu_mxfp8(
    at::Tensor& output,
    at::Tensor& output_scales,
    at::Tensor const& activation,
    at::Tensor const& activation_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    int32_t const* source_tokens,
    bool use_pdl,
    cudaStream_t stream,
    Qwen35GroupedMetadataDevice grouped_metadata = {}) {
  if (source_tokens == nullptr) {
    launch_grouped_w1_silu_mxfp8_shape<false>(
        output, output_scales, activation, activation_scales,
        weight, weight_scales, expert_offsets, scale_offsets,
        source_tokens, use_pdl, stream, grouped_metadata);
  } else {
    launch_grouped_w1_silu_mxfp8_shape<true>(
        output, output_scales, activation, activation_scales,
        weight, weight_scales, expert_offsets, scale_offsets,
        source_tokens, use_pdl, stream, grouped_metadata);
  }
}

template <
    class Id,
    class ElementD,
    bool UsePdl = false,
    bool ExternalShared = false,
    bool SeparateShared = false,
    bool VectorizedPackedLoads = false>
void launch_reduce(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    at::Tensor const& topk_weights,
    at::Tensor const& shared_gate,
    int gate_type,
    int gate_stride,
    int gate_column,
    cudaStream_t stream,
    at::Tensor const* shared_output = nullptr,
    at::Tensor const* shared_weight = nullptr,
    at::Tensor const* shared_weight_scales = nullptr) {
  static_assert(!(UsePdl && ExternalShared));
  static_assert(!(ExternalShared && SeparateShared));
  int const tokens = static_cast<int>(topk_ids.size(0));
  int const n = static_cast<int>(weight.size(1));
  int const k = static_cast<int>(activation.size(1));
  dim3 const grid(n / kReduceTileM, tokens, 1);
  auto const* activation_ptr =
      static_cast<uint8_t const*>(activation.data_ptr());
  auto const* logical_scales_ptr =
      logical_scales.data_ptr<uint8_t>();
  auto const* weight_ptr = weight.data_ptr<uint8_t>();
  auto const* weight_scales_ptr =
      weight_scales.data_ptr<uint8_t>();
  uint8_t const* shared_weight_ptr = nullptr;
  uint8_t const* shared_weight_scales_ptr = nullptr;
  if constexpr (SeparateShared) {
    shared_weight_ptr = shared_weight->data_ptr<uint8_t>();
    shared_weight_scales_ptr =
        shared_weight_scales->data_ptr<uint8_t>();
  }
  auto const* topk_ids_ptr = topk_ids.data_ptr<Id>();
  auto const* topk_weights_ptr =
      topk_weights.data_ptr<float>();
  auto const* shared_gate_ptr = shared_gate.data_ptr();
  ElementD const* shared_output_ptr = nullptr;
  if constexpr (ExternalShared) {
    shared_output_ptr = reinterpret_cast<ElementD const*>(
        shared_output->data_ptr());
  }
  auto* output_ptr =
      reinterpret_cast<ElementD*>(output.data_ptr());
  constexpr int kernel_threads =
      (ExternalShared ? kReduceMaxRoutes - 1 : kReduceMaxRoutes) * 32;
  if constexpr (UsePdl) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(kernel_threads);
    config.dynamicSmemBytes = sizeof(ReduceSharedStorage);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id =
        cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        route_w6a8_reduce_kernel<
            Id, ElementD, true, ExternalShared, SeparateShared,
            VectorizedPackedLoads>,
        activation_ptr,
        logical_scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        shared_weight_ptr,
        shared_weight_scales_ptr,
        topk_ids_ptr,
        topk_weights_ptr,
        shared_gate_ptr,
        shared_output_ptr,
        output_ptr,
        gate_type,
        gate_stride,
        gate_column,
        static_cast<int>(weight.size(0)),
        static_cast<int>(weight.size(2)),
        n,
        k));
  } else {
    route_w6a8_reduce_kernel<
        Id, ElementD, false, ExternalShared, SeparateShared,
        VectorizedPackedLoads>
        <<<grid,
           kernel_threads,
           sizeof(ReduceSharedStorage),
           stream>>>(
            activation_ptr,
            logical_scales_ptr,
            weight_ptr,
            weight_scales_ptr,
            shared_weight_ptr,
            shared_weight_scales_ptr,
            topk_ids_ptr,
            topk_weights_ptr,
            shared_gate_ptr,
            shared_output_ptr,
            output_ptr,
            gate_type,
            gate_stride,
            gate_column,
            static_cast<int>(weight.size(0)),
            static_cast<int>(weight.size(2)),
            n,
            k);
  }
}

}  // namespace direct_moe

void qwen35_grouped_w1_silu_mxfp8_out_cuda(
    at::Tensor& output,
    at::Tensor& output_scales,
    at::Tensor const& activation,
    at::Tensor const& activation_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    bool use_pdl,
    std::optional<at::Tensor> source_tokens_arg) {
  TORCH_CHECK(
      activation.is_cuda() && activation.is_contiguous() &&
          activation.dim() == 2 && activation.size(0) > 0 &&
          activation.size(1) == 2048 &&
          (activation.scalar_type() == at::kByte ||
           activation.scalar_type() == at::kFloat8_e4m3fn),
      "activation must be contiguous CUDA MXFP8 [routes,2048]");
  c10::cuda::CUDAGuard device_guard(activation.device());
  auto const device = activation.device();
  int32_t const* source_tokens_ptr = nullptr;
  int64_t rows = activation.size(0);
  if (source_tokens_arg.has_value()) {
    auto const& source_tokens = *source_tokens_arg;
    TORCH_CHECK(
        source_tokens.is_cuda() && source_tokens.device() == device &&
            source_tokens.scalar_type() == at::kInt &&
            source_tokens.is_contiguous() && source_tokens.dim() == 1 &&
            source_tokens.numel() > 0,
        "source_tokens must be contiguous CUDA int32 [routes]");
    source_tokens_ptr = source_tokens.data_ptr<int32_t>();
    rows = source_tokens.numel();
  }
  TORCH_CHECK(
      expert_offsets.is_cuda() && expert_offsets.device() == device &&
          expert_offsets.scalar_type() == at::kLong &&
          expert_offsets.is_contiguous() &&
          (expert_offsets.numel() == 257 ||
           expert_offsets.numel() == 258) &&
          scale_offsets.is_cuda() && scale_offsets.device() == device &&
          scale_offsets.scalar_type() == at::kLong &&
          scale_offsets.is_contiguous() &&
          scale_offsets.sizes() == expert_offsets.sizes(),
      "expert_offsets and scale_offsets must be CUDA int64 [257] or [258]");
  TORCH_CHECK(
      activation_scales.is_cuda() &&
          activation_scales.device() == device &&
          activation_scales.scalar_type() == at::kByte &&
          activation_scales.is_contiguous(),
      "activation_scales must be contiguous CUDA uint8");
  TORCH_CHECK(
      weight.is_cuda() && weight.device() == device &&
          weight.scalar_type() == at::kByte && weight.is_contiguous() &&
          weight.dim() == 3 &&
          (weight.size(0) == 256 || weight.size(0) == 257) &&
          weight.size(1) == 512 && weight.size(2) == 1536,
      "weight must be canonical packed MXFP6 [256|257,512,1536]");
  TORCH_CHECK(
      expert_offsets.numel() == weight.size(0) + 1,
      "expert offsets must contain E+1 entries");
  TORCH_CHECK(
      weight_scales.is_cuda() && weight_scales.device() == device &&
          weight_scales.scalar_type() == at::kByte &&
          weight_scales.is_contiguous() &&
          weight_scales.numel() >=
              weight.size(0) * 512 * 64,
      "weight_scales is too small for [E,512,64]");
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          output.scalar_type() == at::kByte && output.is_contiguous() &&
          output.sizes() == at::IntArrayRef({rows, 256}),
      "output must be contiguous CUDA uint8 [routes,256]");
  int64_t const max_scale_rows = rows + std::min<int64_t>(rows, 256) * 127;
  TORCH_CHECK(
      output_scales.is_cuda() && output_scales.device() == device &&
          output_scales.scalar_type() == at::kByte &&
          output_scales.is_contiguous() &&
          output_scales.numel() >= max_scale_rows * 8,
      "output_scales is too small for grouped [routes,8] scales");
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(activation.get_device());
  TORCH_CHECK(
      properties.major == 12 && properties.minor == 0,
      "fused grouped W1 requires SM120");
  auto stream = c10::cuda::getCurrentCUDAStream(activation.get_device());
  direct_moe::launch_grouped_w1_silu_mxfp8(
      output,
      output_scales,
      activation,
      activation_scales,
      weight,
      weight_scales,
      expert_offsets,
      scale_offsets,
      source_tokens_ptr,
      use_pdl,
      stream.stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void prepare_grouped_scale_offsets(
    int64_t const* expert_offsets,
    int64_t* scale_offsets,
    int num_experts) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int64_t padded_row = 0;
  for (int expert = 0; expert < num_experts; ++expert) {
    scale_offsets[expert] = padded_row;
    int64_t const rows =
        expert_offsets[expert + 1] - expert_offsets[expert];
    padded_row +=
        (rows + kScaleRowsPerAtom - 1) / kScaleRowsPerAtom *
        kScaleRowsPerAtom;
  }
  scale_offsets[num_experts] = padded_row;
}

__device__ __forceinline__ int64_t grouped_scale_offset(
    int64_t packed_row,
    int k_block,
    int k_blocks) {
  int64_t const row_tile = packed_row / kScaleRowsPerAtom;
  int const row_in_tile =
      static_cast<int>(packed_row % kScaleRowsPerAtom);
  int const k_tiles =
      (k_blocks + kScaleGroupsPerAtom - 1) / kScaleGroupsPerAtom;
  int64_t const tile =
      row_tile * k_tiles + k_block / kScaleGroupsPerAtom;
  return tile * kScaleRowsPerAtom * kScaleGroupsPerAtom +
      (row_in_tile % 32) * 16 +
      (row_in_tile / 32) * 4 +
      k_block % kScaleGroupsPerAtom;
}

template <class Id>
__device__ __forceinline__ int routed_local_expert(
    Id global_expert,
    int32_t const* expert_map,
    int expert_map_size,
    int local_experts) {
  int64_t const global = static_cast<int64_t>(global_expert);
  if (global < 0) {
    return -1;
  }
  int local = static_cast<int>(global);
  if (expert_map != nullptr) {
    if (global >= expert_map_size) {
      return -1;
    }
    local = expert_map[global];
  }
  return local >= 0 && local < local_experts ? local : -1;
}

template <class Id>
__global__ void histogram_routed_experts_kernel(
    Id const* topk_ids,
    int32_t const* expert_map,
    int32_t* cursors,
    int64_t* expert_offsets,
    int64_t* scale_offsets,
    int routes,
    int expert_map_size,
    int routed_experts,
    int total_experts,
    int shared_rows) {
  __shared__ int32_t local_counts[kMaxMoEExperts];
  __shared__ int64_t warp_row_prefix[16];
  __shared__ int64_t warp_scale_row_prefix[16];
  for (int expert = threadIdx.x;
       expert < total_experts;
       expert += blockDim.x) {
    local_counts[expert] = 0;
  }
  __syncthreads();

  for (int route = threadIdx.x; route < routes; route += blockDim.x) {
    int const expert = routed_local_expert(
        topk_ids[route], expert_map, expert_map_size, routed_experts);
    if (expert >= 0) {
      atomicAdd(local_counts + expert, 1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && shared_rows > 0) {
    local_counts[routed_experts] = shared_rows;
  }
  __syncthreads();

  int const thread = static_cast<int>(threadIdx.x);
  int const lane = thread & 31;
  int const warp = thread >> 5;
  int const warp_count = (total_experts + 31) / 32;
  int64_t const rows =
      thread < total_experts ? local_counts[thread] : 0;
  int64_t row_prefix = rows;
  int64_t scale_row_prefix =
      (rows + kScaleRowsPerAtom - 1) &
      -static_cast<int64_t>(kScaleRowsPerAtom);

#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    int64_t const row_add =
        __shfl_up_sync(0xffffffffu, row_prefix, offset);
    int64_t const scale_row_add =
        __shfl_up_sync(0xffffffffu, scale_row_prefix, offset);
    if (lane >= offset) {
      row_prefix += row_add;
      scale_row_prefix += scale_row_add;
    }
  }
  if (lane == 31) {
    warp_row_prefix[warp] = row_prefix;
    warp_scale_row_prefix[warp] = scale_row_prefix;
  }
  __syncthreads();

  if (warp == 0) {
    int64_t warp_rows =
        lane < warp_count ? warp_row_prefix[lane] : 0;
    int64_t warp_scale_rows =
        lane < warp_count ? warp_scale_row_prefix[lane] : 0;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      int64_t const row_add =
          __shfl_up_sync(0xffffffffu, warp_rows, offset);
      int64_t const scale_row_add =
          __shfl_up_sync(0xffffffffu, warp_scale_rows, offset);
      if (lane >= offset) {
        warp_rows += row_add;
        warp_scale_rows += scale_row_add;
      }
    }
    if (lane < warp_count) {
      warp_row_prefix[lane] = warp_rows;
      warp_scale_row_prefix[lane] = warp_scale_rows;
    }
  }
  __syncthreads();

  if (thread < total_experts) {
    int64_t const prior_warp_rows =
        warp == 0 ? 0 : warp_row_prefix[warp - 1];
    int64_t const prior_warp_scale_rows =
        warp == 0 ? 0 : warp_scale_row_prefix[warp - 1];
    int64_t const row_offset =
        prior_warp_rows + row_prefix - rows;
    int64_t const padded_rows =
        (rows + kScaleRowsPerAtom - 1) &
        -static_cast<int64_t>(kScaleRowsPerAtom);
    int64_t const scale_row_offset =
        prior_warp_scale_rows + scale_row_prefix - padded_rows;
    expert_offsets[thread] = row_offset;
    scale_offsets[thread] = scale_row_offset;
    cursors[thread] = static_cast<int32_t>(row_offset);
    if (thread == total_experts - 1) {
      expert_offsets[total_experts] =
          prior_warp_rows + row_prefix;
      scale_offsets[total_experts] =
          prior_warp_scale_rows + scale_row_prefix;
    }
  }
}

template <class Id, bool UsePdl = false>
__global__ void gather_routed_mxfp8_kernel(
    uint8_t const* activation,
    uint8_t const* logical_scales,
    Id const* topk_ids,
    int32_t const* expert_map,
    int32_t* cursors,
    int64_t const* expert_offsets,
    int64_t const* scale_offsets,
    uint8_t* permuted_activation,
    uint8_t* packed_scales,
    int32_t* inverse_permutation,
    int routes,
    int topk,
    int k,
    int expert_map_size,
    int routed_experts,
    int shared_rows) {
  int const work = blockIdx.x;
  __shared__ int destination;
  __shared__ int expert;
  __shared__ int source_row;
  if (threadIdx.x == 0) {
    if (work < routes) {
      expert = routed_local_expert(
          topk_ids[work], expert_map, expert_map_size, routed_experts);
      destination =
          expert >= 0 ? atomicAdd(cursors + expert, 1) : routes + shared_rows;
      inverse_permutation[work] = destination;
      source_row = work / topk;
    } else {
      expert = routed_experts;
      source_row = work - routes;
      destination = expert_offsets[expert] + source_row;
    }
  }
  __syncthreads();
  if (expert < 0) {
    if constexpr (UsePdl) {
      if (threadIdx.x == 0) {
        cutlass::arch::launch_dependent_grids();
      }
    }
    return;
  }

  auto const* source_values = reinterpret_cast<uint4 const*>(
      activation + static_cast<int64_t>(source_row) * k);
  auto* destination_values = reinterpret_cast<uint4*>(
      permuted_activation + static_cast<int64_t>(destination) * k);
  for (int vector = threadIdx.x; vector < k / 16; vector += blockDim.x) {
    destination_values[vector] = source_values[vector];
  }

  int const k_blocks = k / kMxScaleVectorSize;
  int64_t const local_row =
      destination - expert_offsets[expert];
  int64_t const packed_row =
      scale_offsets[expert] + local_row;
  auto const* source_scales =
      logical_scales + static_cast<int64_t>(source_row) * k_blocks;
  for (int k_block = threadIdx.x;
       k_block < k_blocks;
       k_block += blockDim.x) {
    packed_scales[grouped_scale_offset(
        packed_row, k_block, k_blocks)] = source_scales[k_block];
  }
  if constexpr (UsePdl) {
    __syncthreads();
    if (threadIdx.x == 0) {
      cutlass::arch::launch_dependent_grids();
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
route_mxfp8_out_cuda(
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& topk_ids,
    std::optional<at::Tensor> expert_map_arg,
    int64_t local_experts,
    at::Tensor& permuted_activation,
    bool include_shared) {
  TORCH_CHECK(
      activation.is_cuda() &&
          (activation.scalar_type() == at::kFloat8_e4m3fn ||
           activation.scalar_type() == at::kByte) &&
          activation.is_contiguous() && activation.dim() == 2,
      "activation must be contiguous CUDA float8_e4m3fn/uint8 [M,K]");
  TORCH_CHECK(
      logical_scales.is_cuda() &&
          logical_scales.device() == activation.device() &&
          logical_scales.scalar_type() == at::kByte &&
          logical_scales.is_contiguous() && logical_scales.dim() == 2,
      "logical_scales must be contiguous CUDA uint8 [M,K/32]");
  TORCH_CHECK(
      topk_ids.is_cuda() &&
          topk_ids.device() == activation.device() &&
          (topk_ids.scalar_type() == at::kInt ||
           topk_ids.scalar_type() == at::kLong) &&
          topk_ids.is_contiguous() && topk_ids.dim() == 2,
      "topk_ids must be contiguous CUDA int32/int64 [M,topk]");
  TORCH_CHECK(
      local_experts > (include_shared ? 1 : 0) &&
          local_experts <= kMaxMoEExperts,
      "local_experts must be in [1, 257] and include one trailing shared "
      "expert when include_shared is true");

  int64_t const tokens = activation.size(0);
  int64_t const k = activation.size(1);
  int64_t const topk = topk_ids.size(1);
  int64_t const routes = tokens * topk;
  int64_t const shared_rows = include_shared ? tokens : 0;
  int64_t const total_rows = routes + shared_rows;
  int64_t const routed_experts =
      local_experts - (include_shared ? 1 : 0);
  TORCH_CHECK(
      tokens > 0 && topk > 0 &&
          routes <= std::numeric_limits<int>::max() &&
          total_rows <= std::numeric_limits<int>::max(),
      "number of routed/shared rows must be positive and fit int32");
  TORCH_CHECK(
      k > 0 && k % kMxScaleVectorSize == 0 &&
          k <= std::numeric_limits<int>::max(),
      "activation K must be a positive multiple of 32 and fit int32");
  TORCH_CHECK(
      topk_ids.size(0) == tokens &&
          logical_scales.size(0) == tokens &&
          logical_scales.size(1) == k / kMxScaleVectorSize,
      "activation, logical_scales, and topk_ids shapes are inconsistent");
  TORCH_CHECK(
      permuted_activation.is_cuda() &&
          permuted_activation.device() == activation.device() &&
          permuted_activation.scalar_type() == activation.scalar_type() &&
          permuted_activation.is_contiguous() &&
          permuted_activation.dim() == 2 &&
          permuted_activation.size(0) == total_rows &&
          permuted_activation.size(1) == k,
      "permuted_activation must contain routed rows and, when requested, "
      "one trailing shared-expert row per token");

  int32_t const* expert_map = nullptr;
  int expert_map_size = 0;
  if (expert_map_arg.has_value()) {
    at::Tensor const& map = *expert_map_arg;
    TORCH_CHECK(
        map.is_cuda() && map.device() == activation.device() &&
            map.scalar_type() == at::kInt &&
            map.is_contiguous() && map.dim() == 1 &&
            map.numel() <= std::numeric_limits<int>::max(),
        "expert_map must be a contiguous CUDA int32 vector");
    expert_map = map.data_ptr<int32_t>();
    expert_map_size = static_cast<int>(map.numel());
  }

  c10::cuda::CUDAGuard device_guard(activation.device());
  auto byte_options = activation.options().dtype(at::kByte);
  auto int_options = activation.options().dtype(at::kInt);
  auto long_options = activation.options().dtype(at::kLong);
  auto cursors_tensor = at::empty(
      {local_experts}, int_options);
  auto expert_offsets = at::empty(
      {local_experts + 1}, long_options);
  auto scale_offsets = at::empty(
      {local_experts + 1}, long_options);
  auto inverse_permutation = at::empty(
      {routes}, int_options);
  int64_t const scale_rows =
      total_rows + local_experts * (kScaleRowsPerAtom - 1);
  int64_t const k_blocks = k / kMxScaleVectorSize;
  int64_t const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  auto packed_scales = at::empty(
      {scale_rows * packed_k_blocks}, byte_options);

  auto* cursors = cursors_tensor.data_ptr<int32_t>();
  int const device_index = activation.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  C10_CUDA_CHECK(cudaMemsetAsync(
      packed_scales.data_ptr<uint8_t>(), kUe8m0One,
      static_cast<size_t>(packed_scales.numel()), stream.stream()));
  constexpr int threads = 256;
  constexpr int routing_threads = 512;
  if (topk_ids.scalar_type() == at::kInt) {
    histogram_routed_experts_kernel<int32_t>
        <<<1, routing_threads, 0, stream.stream()>>>(
            topk_ids.data_ptr<int32_t>(),
            expert_map,
            cursors,
            expert_offsets.data_ptr<int64_t>(),
            scale_offsets.data_ptr<int64_t>(),
            static_cast<int>(routes),
            expert_map_size,
            static_cast<int>(routed_experts),
            static_cast<int>(local_experts),
            static_cast<int>(shared_rows));
  } else {
    histogram_routed_experts_kernel<int64_t>
        <<<1, routing_threads, 0, stream.stream()>>>(
            topk_ids.data_ptr<int64_t>(),
            expert_map,
            cursors,
            expert_offsets.data_ptr<int64_t>(),
            scale_offsets.data_ptr<int64_t>(),
            static_cast<int>(routes),
            expert_map_size,
            static_cast<int>(routed_experts),
            static_cast<int>(local_experts),
            static_cast<int>(shared_rows));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (topk_ids.scalar_type() == at::kInt) {
    gather_routed_mxfp8_kernel<int32_t>
        <<<static_cast<int>(total_rows), threads, 0, stream.stream()>>>(
            static_cast<uint8_t const*>(activation.data_ptr()),
            logical_scales.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int32_t>(),
            expert_map,
            cursors,
            expert_offsets.data_ptr<int64_t>(),
            scale_offsets.data_ptr<int64_t>(),
            static_cast<uint8_t*>(permuted_activation.data_ptr()),
            packed_scales.data_ptr<uint8_t>(),
            inverse_permutation.data_ptr<int32_t>(),
            static_cast<int>(routes),
            static_cast<int>(topk),
            static_cast<int>(k),
            expert_map_size,
            static_cast<int>(routed_experts),
            static_cast<int>(shared_rows));
  } else {
    gather_routed_mxfp8_kernel<int64_t>
        <<<static_cast<int>(total_rows), threads, 0, stream.stream()>>>(
            static_cast<uint8_t const*>(activation.data_ptr()),
            logical_scales.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int64_t>(),
            expert_map,
            cursors,
            expert_offsets.data_ptr<int64_t>(),
            scale_offsets.data_ptr<int64_t>(),
            static_cast<uint8_t*>(permuted_activation.data_ptr()),
            packed_scales.data_ptr<uint8_t>(),
            inverse_permutation.data_ptr<int32_t>(),
            static_cast<int>(routes),
            static_cast<int>(topk),
            static_cast<int>(k),
            expert_map_size,
            static_cast<int>(routed_experts),
            static_cast<int>(shared_rows));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      packed_scales,
      expert_offsets,
      scale_offsets,
      inverse_permutation};
}

void qwen35_route_mxfp8_out_cuda(
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& topk_ids,
    at::Tensor& permuted_activation,
    at::Tensor& packed_scales,
    at::Tensor& expert_cursors,
    at::Tensor& expert_offsets,
    at::Tensor& scale_offsets,
    at::Tensor& inverse_permutation,
    bool include_shared) {
  TORCH_CHECK(
      activation.is_cuda() &&
          (activation.scalar_type() == at::kFloat8_e4m3fn ||
           activation.scalar_type() == at::kByte) &&
          activation.is_contiguous() &&
          activation.dim() == 2 &&
          activation.size(0) > 0 &&
          activation.size(0) <=
              std::numeric_limits<int>::max() /
                  (qwen35_router::kTopK +
                   static_cast<int>(include_shared)) &&
          activation.size(1) == qwen35_router::kHidden,
      "activation must be contiguous CUDA MXFP8 [B,2048] and B*8 must "
      "fit int32");
  c10::cuda::CUDAGuard device_guard(activation.device());
  auto const device = activation.device();
  int64_t const tokens = activation.size(0);
  int64_t const routes =
      tokens * qwen35_router::kTopK;
  int64_t const shared_rows = include_shared ? tokens : 0;
  int64_t const total_rows = routes + shared_rows;
  int64_t const total_experts =
      qwen35_router::kRoutedExperts +
      static_cast<int64_t>(include_shared);
  TORCH_CHECK(
      logical_scales.is_cuda() &&
          logical_scales.device() == device &&
          logical_scales.scalar_type() == at::kByte &&
          logical_scales.is_contiguous() &&
          logical_scales.sizes() ==
              at::IntArrayRef(
                  {tokens, qwen35_router::kGroupsPerRow}),
      "logical_scales must be contiguous CUDA uint8 [B,64]");
  TORCH_CHECK(
      topk_ids.is_cuda() &&
          topk_ids.device() == device &&
          topk_ids.scalar_type() == at::kInt &&
          topk_ids.is_contiguous() &&
          topk_ids.sizes() ==
              at::IntArrayRef(
                  {tokens, qwen35_router::kTopK}),
      "topk_ids must be contiguous CUDA int32 [B,8]");
  TORCH_CHECK(
      permuted_activation.is_cuda() &&
          permuted_activation.device() == device &&
          permuted_activation.scalar_type() ==
              activation.scalar_type() &&
          permuted_activation.is_contiguous() &&
          permuted_activation.sizes() ==
              at::IntArrayRef(
                  {total_rows, qwen35_router::kHidden}),
      "permuted_activation must contain all routed and optional shared "
      "rows");
  int64_t const active_experts =
      std::min<int64_t>(
          routes, qwen35_router::kRoutedExperts) +
      static_cast<int64_t>(include_shared);
  int64_t const max_scale_rows =
      total_rows +
      active_experts * (kScaleRowsPerAtom - 1);
  TORCH_CHECK(
      packed_scales.is_cuda() &&
          packed_scales.device() == device &&
          packed_scales.scalar_type() == at::kByte &&
          packed_scales.is_contiguous() &&
          packed_scales.numel() >=
              max_scale_rows *
                  qwen35_router::kGroupsPerRow,
      "packed_scales is too small for routed SM120 scale packing");
  TORCH_CHECK(
      expert_cursors.is_cuda() &&
          expert_cursors.device() == device &&
          expert_cursors.scalar_type() == at::kInt &&
          expert_cursors.is_contiguous() &&
          expert_cursors.numel() == total_experts,
      "expert_cursors must have one contiguous CUDA int32 entry per "
      "expert");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.device() == device &&
          expert_offsets.scalar_type() == at::kLong &&
          expert_offsets.is_contiguous() &&
          expert_offsets.numel() == total_experts + 1,
      "expert_offsets must have one contiguous CUDA int64 entry per "
      "expert plus a sentinel");
  TORCH_CHECK(
      scale_offsets.is_cuda() &&
          scale_offsets.device() == device &&
          scale_offsets.scalar_type() == at::kLong &&
          scale_offsets.is_contiguous() &&
          scale_offsets.sizes() == expert_offsets.sizes(),
      "scale_offsets must match expert_offsets");
  TORCH_CHECK(
      inverse_permutation.is_cuda() &&
          inverse_permutation.device() == device &&
          inverse_permutation.scalar_type() == at::kInt &&
          inverse_permutation.is_contiguous() &&
          inverse_permutation.numel() == routes,
      "inverse_permutation must be contiguous CUDA int32 [B*8]");
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(activation.get_device());
  TORCH_CHECK(
      properties.major == 12 && properties.minor == 0,
      "Qwen3.5 preallocated routing requires SM120");

  auto stream = c10::cuda::getCurrentCUDAStream(
      activation.get_device());
  histogram_routed_experts_kernel<int32_t>
      <<<1, 512, 0, stream.stream()>>>(
          topk_ids.data_ptr<int32_t>(),
          nullptr,
          expert_cursors.data_ptr<int32_t>(),
          expert_offsets.data_ptr<int64_t>(),
          scale_offsets.data_ptr<int64_t>(),
          static_cast<int>(routes),
          0,
          qwen35_router::kRoutedExperts,
          static_cast<int>(total_experts),
          static_cast<int>(shared_rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gather_routed_mxfp8_kernel<int32_t, true>
      <<<static_cast<int>(total_rows),
         256,
         0,
         stream.stream()>>>(
          static_cast<uint8_t const*>(
              activation.data_ptr()),
          logical_scales.data_ptr<uint8_t>(),
          topk_ids.data_ptr<int32_t>(),
          nullptr,
          expert_cursors.data_ptr<int32_t>(),
          expert_offsets.data_ptr<int64_t>(),
          scale_offsets.data_ptr<int64_t>(),
          static_cast<uint8_t*>(
              permuted_activation.data_ptr()),
          packed_scales.data_ptr<uint8_t>(),
          inverse_permutation.data_ptr<int32_t>(),
          static_cast<int>(routes),
          qwen35_router::kTopK,
          qwen35_router::kHidden,
          0,
          qwen35_router::kRoutedExperts,
          static_cast<int>(shared_rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void pack_grouped_scales_kernel(
    uint8_t const* logical,
    uint8_t* packed,
    int64_t const* expert_offsets,
    int64_t const* scale_offsets,
    int64_t rows,
    int k_blocks,
    int num_experts) {
  int64_t const linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const elements = rows * k_blocks;
  if (linear >= elements) {
    return;
  }

  int64_t const row = linear / k_blocks;
  int const k_block = static_cast<int>(linear - row * k_blocks);
  int lower = 0;
  int upper = num_experts;
  while (lower < upper) {
    int const middle = lower + (upper - lower) / 2;
    if (row < expert_offsets[middle + 1]) {
      upper = middle;
    } else {
      lower = middle + 1;
    }
  }
  if (lower == num_experts) {
    return;
  }

  int64_t const local_row = row - expert_offsets[lower];
  int64_t const packed_row = scale_offsets[lower] + local_row;
  int64_t const destination =
      grouped_scale_offset(packed_row, k_block, k_blocks);
  packed[destination] = logical[linear];
}

std::tuple<at::Tensor, at::Tensor> pack_grouped_scales_cuda(
    at::Tensor const& logical,
    at::Tensor const& expert_offsets) {
  TORCH_CHECK(
      logical.is_cuda() && logical.scalar_type() == at::kByte &&
          logical.is_contiguous() && logical.dim() == 2,
      "logical scales must be a contiguous CUDA uint8 [M,K/32] tensor");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.device() == logical.device() &&
          expert_offsets.scalar_type() == at::kLong &&
          expert_offsets.is_contiguous() && expert_offsets.dim() == 1,
      "expert_offsets must be a contiguous CUDA int64 tensor");
  TORCH_CHECK(
      expert_offsets.numel() >= 2,
      "expert_offsets must contain at least two entries");

  int64_t const rows = logical.size(0);
  int64_t const k_blocks = logical.size(1);
  int64_t const num_experts = expert_offsets.numel() - 1;
  TORCH_CHECK(
      rows > 0 && rows <= std::numeric_limits<int>::max(),
      "scale rows must be positive and fit int32");
  TORCH_CHECK(
      k_blocks > 0 && k_blocks <= std::numeric_limits<int>::max(),
      "K/32 must be positive and fit int32");
  TORCH_CHECK(
      num_experts <= std::numeric_limits<int>::max(),
      "number of experts must fit int32");

  c10::cuda::CUDAGuard device_guard(logical.device());
  int64_t const active_expert_bound = std::min(rows, num_experts);
  int64_t const max_padded_rows =
      rows + active_expert_bound * (kScaleRowsPerAtom - 1);
  int64_t const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  TORCH_CHECK(
      max_padded_rows <=
          std::numeric_limits<int64_t>::max() / packed_k_blocks,
      "packed scale size overflows int64");
  auto packed = at::empty(
      {max_padded_rows * packed_k_blocks}, logical.options());
  auto scale_offsets = at::empty(
      {num_experts + 1}, expert_offsets.options());

  int const device_index = logical.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  C10_CUDA_CHECK(cudaMemsetAsync(
      packed.data_ptr<uint8_t>(), kUe8m0One,
      static_cast<size_t>(packed.numel()), stream.stream()));
  prepare_grouped_scale_offsets<<<1, 1, 0, stream.stream()>>>(
      expert_offsets.data_ptr<int64_t>(),
      scale_offsets.data_ptr<int64_t>(),
      static_cast<int>(num_experts));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int64_t const elements = rows * k_blocks;
  int64_t const blocks =
      (elements + kScaleThreads - 1) / kScaleThreads;
  TORCH_CHECK(
      blocks <= std::numeric_limits<int>::max(),
      "grouped scale launch grid is too large");
  pack_grouped_scales_kernel
      <<<static_cast<int>(blocks), kScaleThreads, 0, stream.stream()>>>(
          logical.data_ptr<uint8_t>(),
          packed.data_ptr<uint8_t>(),
          expert_offsets.data_ptr<int64_t>(),
          scale_offsets.data_ptr<int64_t>(),
          rows,
          static_cast<int>(k_blocks),
          static_cast<int>(num_experts));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scale_offsets};
}

template <class Source>
__device__ __forceinline__ float source_round(float value) {
  return static_cast<float>(static_cast<Source>(value));
}

__device__ __forceinline__ uint16_t quantize_fp8_pair(
    float first,
    float second) {
  return __nv_cvt_float2_to_fp8x2(
      make_float2(first, second), __NV_SATFINITE, __NV_E4M3);
}

template <class T>
at::Tensor qwen35_metadata_tensor(
    int64_t count,
    at::Tensor const& anchor) {
  static_assert(std::is_trivially_copyable_v<T>);
  return at::empty(
      {count * static_cast<int64_t>(sizeof(T))},
      anchor.options().dtype(at::kByte));
}

struct Qwen35GroupedMetadataStorage {
  int group_count;
  at::Tensor problem_shapes;
  at::Tensor ptr_a;
  at::Tensor ptr_b;
  at::Tensor ptr_sfa;
  at::Tensor ptr_sfb;
  at::Tensor ptr_d;
  at::Tensor stride_a;
  at::Tensor stride_b;
  at::Tensor stride_c;
  at::Tensor stride_d;
  at::Tensor layout_sfa;
  at::Tensor layout_sfb;
  at::Tensor work_tiles;
  at::Tensor work_tile_count;

  Qwen35GroupedMetadataStorage(
      int count,
      at::Tensor const& anchor)
      : group_count(count),
        problem_shapes(qwen35_metadata_tensor<Qwen35GroupedProblem>(
            count, anchor)),
        ptr_a(qwen35_metadata_tensor<
              typename Qwen35GroupedKernel::ElementA const*>(
            count, anchor)),
        ptr_b(qwen35_metadata_tensor<
              typename Qwen35GroupedKernel::ElementB const*>(
            count, anchor)),
        ptr_sfa(qwen35_metadata_tensor<
                typename Qwen35GroupedKernel::ElementSF const*>(
            count, anchor)),
        ptr_sfb(qwen35_metadata_tensor<
                typename Qwen35GroupedKernel::ElementSF const*>(
            count, anchor)),
        ptr_d(qwen35_metadata_tensor<
              typename Qwen35GroupedKernel::ElementD*>(
            count, anchor)),
        stride_a(qwen35_metadata_tensor<
                 typename Qwen35GroupedKernel::StrideA>(count, anchor)),
        stride_b(qwen35_metadata_tensor<
                 typename Qwen35GroupedKernel::StrideB>(count, anchor)),
        stride_c(qwen35_metadata_tensor<
                 typename Qwen35GroupedKernel::StrideC>(count, anchor)),
        stride_d(qwen35_metadata_tensor<
                 typename Qwen35GroupedKernel::StrideD>(count, anchor)),
        layout_sfa(qwen35_metadata_tensor<
                   typename Qwen35GroupedKernel::LayoutSFA>(count, anchor)),
        layout_sfb(qwen35_metadata_tensor<
                   typename Qwen35GroupedKernel::LayoutSFB>(count, anchor)),
        work_tiles(at::empty(
            {std::max<int64_t>(1, anchor.size(0) * 8), 4},
            anchor.options().dtype(at::kInt))),
        work_tile_count(at::empty(
            {1}, anchor.options().dtype(at::kInt))) {}

  Qwen35GroupedMetadataDevice device_view(
      at::Tensor const& activation,
      at::Tensor const& activation_scales,
      at::Tensor const& weight,
      at::Tensor const& weight_scales,
      at::Tensor const& expert_offsets,
      at::Tensor const& scale_offsets,
      at::Tensor& output) {
    int const n = static_cast<int>(weight.size(1));
    int const k = static_cast<int>(activation.size(1));
    bool const explicit_tiles = grouped_explicit_tiles();
    return {
        static_cast<uint8_t const*>(activation.data_ptr()),
        activation_scales.data_ptr<uint8_t>(),
        weight.data_ptr<uint8_t>(),
        weight_scales.data_ptr<uint8_t>(),
        expert_offsets.data_ptr<int64_t>(),
        scale_offsets.data_ptr<int64_t>(),
        reinterpret_cast<typename Qwen35GroupedKernel::ElementD*>(
            output.data_ptr()),
        reinterpret_cast<Qwen35GroupedProblem*>(
            problem_shapes.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::ElementA const**>(
            ptr_a.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::ElementB const**>(
            ptr_b.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::ElementSF const**>(
            ptr_sfa.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::ElementSF const**>(
            ptr_sfb.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::ElementD**>(
            ptr_d.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::StrideA*>(
            stride_a.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::StrideB*>(
            stride_b.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::StrideC*>(
            stride_c.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::StrideD*>(
            stride_d.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::LayoutSFA*>(
            layout_sfa.data_ptr()),
        reinterpret_cast<typename Qwen35GroupedKernel::LayoutSFB*>(
            layout_sfb.data_ptr()),
        explicit_tiles ? work_tiles.data_ptr<int32_t>() : nullptr,
        explicit_tiles ? work_tile_count.data_ptr<int32_t>() : nullptr,
        static_cast<int>(work_tiles.size(0)),
        group_count,
        n,
        k};
  }
};

void launch_qwen35_grouped_from_metadata(
    at::Tensor& output,
    at::Tensor const& activation,
    Qwen35GroupedMetadataStorage& metadata,
    bool use_pdl) {
  using ProblemShape = mxfp6_gemm::grouped::ProblemShape;
  using GemmKernel = typename Qwen35GroupedKernel::GemmKernel;
  using Gemm = typename Qwen35GroupedKernel::Gemm;
  ProblemShape problem{
      metadata.group_count,
      reinterpret_cast<Qwen35GroupedProblem*>(
          metadata.problem_shapes.data_ptr()),
      nullptr};
  typename GemmKernel::MainloopArguments mainloop{
      reinterpret_cast<typename Qwen35GroupedKernel::ElementA const**>(
          metadata.ptr_a.data_ptr()),
      reinterpret_cast<typename Qwen35GroupedKernel::StrideA*>(
          metadata.stride_a.data_ptr()),
      reinterpret_cast<typename Qwen35GroupedKernel::ElementB const**>(
          metadata.ptr_b.data_ptr()),
      reinterpret_cast<typename Qwen35GroupedKernel::StrideB*>(
          metadata.stride_b.data_ptr()),
      reinterpret_cast<typename Qwen35GroupedKernel::ElementSF const**>(
          metadata.ptr_sfa.data_ptr()),
      reinterpret_cast<typename Qwen35GroupedKernel::LayoutSFA*>(
          metadata.layout_sfa.data_ptr()),
      reinterpret_cast<typename Qwen35GroupedKernel::ElementSF const**>(
          metadata.ptr_sfb.data_ptr()),
      reinterpret_cast<typename Qwen35GroupedKernel::LayoutSFB*>(
          metadata.layout_sfb.data_ptr()),
      static_cast<int32_t>(output.size(1)),
      128,
      static_cast<int32_t>(activation.size(1)),
      grouped_fixed_tensormaps() && output.size(0) <= 1024};
  typename GemmKernel::EpilogueArguments epilogue{
      {1.0f, 0.0f},
      nullptr,
      reinterpret_cast<typename Qwen35GroupedKernel::StrideC*>(
          metadata.stride_c.data_ptr()),
      reinterpret_cast<typename Qwen35GroupedKernel::ElementD**>(
          metadata.ptr_d.data_ptr()),
      reinterpret_cast<typename Qwen35GroupedKernel::StrideD*>(
          metadata.stride_d.data_ptr())};
  int const device_index = activation.get_device();
  int const sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          device_index);
  constexpr bool kDualCtaPersistent =
      GemmKernel::MaxThreadsPerBlock <= 256;
  int const persistent_blocks =
      kDualCtaPersistent && grouped_dual_cta_persistent() ? 2 : 1;
  cutlass::KernelHardwareInfo hw_info{
      device_index, sm_count * persistent_blocks};
  typename GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = grouped_raster_along_n()
      ? GemmKernel::TileScheduler::RasterOrderOptions::AlongN
      : GemmKernel::TileScheduler::RasterOrderOptions::AlongM;
  scheduler.max_swizzle_size = grouped_max_swizzle();
  if (grouped_explicit_tiles()) {
    scheduler.explicit_work_tiles =
        metadata.work_tiles.data_ptr<int32_t>();
    scheduler.explicit_work_tile_count =
        metadata.work_tile_count.data_ptr<int32_t>();
    scheduler.explicit_max_work_tiles =
        static_cast<uint32_t>(metadata.work_tiles.size(0));
  }
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      problem,
      mainloop,
      epilogue,
      hw_info,
      scheduler};
  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Qwen3.5 grouped MXFP6 can_implement failed: ",
      cutlassGetStatusString(status));
  size_t const workspace_bytes = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_bytes)},
        activation.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  status = gemm.initialize(arguments, workspace_ptr, stream.stream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Qwen3.5 grouped MXFP6 initialize failed: ",
      cutlassGetStatusString(status));
  status = gemm.run(stream.stream(), nullptr, use_pdl);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Qwen3.5 grouped MXFP6 launch failed: ",
      cutlassGetStatusString(status));
}

template <class Source, bool UsePdl = false>
__global__ void silu_and_mul_mxfp8_grouped_kernel(
    Source const* input,
    uint8_t* output,
    uint8_t* scales,
    int64_t const* expert_offsets,
    int64_t const* scale_offsets,
    int groups_per_row,
    int64_t total_groups,
    int num_experts,
    bool dense_packed,
    Qwen35GroupedMetadataDevice grouped_metadata) {
  if constexpr (UsePdl) {
    cutlass::arch::wait_on_dependent_grids();
  }
  __shared__ int32_t work_tile_cursor;
  if (blockIdx.x == 0 && grouped_metadata.work_tiles != nullptr) {
    if (threadIdx.x == 0) {
      work_tile_cursor = 0;
    }
    __syncthreads();
    int const group = static_cast<int>(threadIdx.x);
    if (group < num_experts) {
      int64_t const rows =
          expert_offsets[group + 1] - expert_offsets[group];
      grouped_metadata.write_work_tiles(
          group,
          static_cast<int>(rows),
          &work_tile_cursor);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      *grouped_metadata.work_tile_count = work_tile_cursor;
    }
  }
  int const metadata_group =
      static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  grouped_metadata.write(metadata_group);
  int const thread_in_group =
      threadIdx.x % kQuantThreadsPerGroup;
  int const group_in_block =
      threadIdx.x / kQuantThreadsPerGroup;
  int64_t const group =
      static_cast<int64_t>(blockIdx.x) * kQuantGroupsPerBlock +
      group_in_block;
  bool const valid = group < total_groups;

  int64_t const row = group / groups_per_row;
  int const k_group = static_cast<int>(
      group - row * groups_per_row);
  int const output_features = groups_per_row * kMxScaleVectorSize;
  int64_t const input_row_offset =
      row * output_features * 2;
  int64_t const value_offset =
      group * kMxScaleVectorSize +
      thread_in_group * kQuantElementsPerThread;

  float values[kQuantElementsPerThread]{};
  if (valid) {
#pragma unroll
    for (int index = 0; index < kQuantElementsPerThread; ++index) {
      int64_t const column =
          static_cast<int64_t>(k_group) * kMxScaleVectorSize +
          thread_in_group * kQuantElementsPerThread + index;
      float gate = static_cast<float>(
          input[input_row_offset + column]);
      float const up = static_cast<float>(
          input[input_row_offset + output_features + column]);
      gate = source_round<Source>(
          gate / (1.0f + expf(-gate)));
      values[index] = source_round<Source>(gate * up);
    }
  }

  float absmax = 0.0f;
#pragma unroll
  for (int index = 0; index < kQuantElementsPerThread; ++index) {
    absmax = fmaxf(absmax, fabsf(values[index]));
  }
  absmax = fmaxf(
      absmax,
      __shfl_xor_sync(
          0xffffffffu, absmax, 1, kQuantThreadsPerGroup));
  absmax = fmaxf(
      absmax,
      __shfl_xor_sync(
          0xffffffffu, absmax, 2, kQuantThreadsPerGroup));

  float inverse_scale = 1.0f;
  uint8_t scale_code = 0x7f;
  if (thread_in_group == 0) {
    float const raw_scale = fmaxf(absmax / 448.0f, 1.0e-30f);
    uint32_t scale_bits = __float_as_uint(raw_scale);
    scale_bits = (scale_bits + 0x007fffffu) & 0x7f800000u;
    scale_code = static_cast<uint8_t>(scale_bits >> 23);
    inverse_scale = 1.0f / __uint_as_float(scale_bits);
  }
  inverse_scale = __shfl_sync(
      0xffffffffu, inverse_scale, 0, kQuantThreadsPerGroup);

  if (valid && thread_in_group == 0) {
    if (dense_packed) {
      scales[grouped_scale_offset(
          row, k_group, groups_per_row)] = scale_code;
    } else if (scale_offsets == nullptr) {
      scales[group] = scale_code;
    } else {
      int lower = 0;
      int upper = num_experts;
      while (lower < upper) {
        int const middle = lower + (upper - lower) / 2;
        if (row < expert_offsets[middle + 1]) {
          upper = middle;
        } else {
          lower = middle + 1;
        }
      }
      if (lower < num_experts) {
        int64_t const packed_row =
            scale_offsets[lower] + row - expert_offsets[lower];
        scales[grouped_scale_offset(
            packed_row, k_group, groups_per_row)] = scale_code;
      }
    }
  }

  uint16_t pairs[kQuantElementsPerThread / 2];
#pragma unroll
  for (int index = 0; index < kQuantElementsPerThread; index += 2) {
    pairs[index / 2] = quantize_fp8_pair(
        values[index] * inverse_scale,
        values[index + 1] * inverse_scale);
  }
  if (valid) {
    uint2 packed{
        static_cast<uint32_t>(pairs[0]) |
            (static_cast<uint32_t>(pairs[1]) << 16),
        static_cast<uint32_t>(pairs[2]) |
            (static_cast<uint32_t>(pairs[3]) << 16)};
    reinterpret_cast<uint2*>(output + value_offset)[0] = packed;
  }
  if constexpr (UsePdl) {
    __syncthreads();
    if (threadIdx.x == 0) {
      cutlass::arch::launch_dependent_grids();
    }
  }
}

template <class Source>
void launch_silu_and_mul_mxfp8_grouped(
    Source const* input,
    uint8_t* output,
    uint8_t* scales,
    int64_t const* expert_offsets,
    int64_t const* scale_offsets,
    int groups_per_row,
    int64_t total_groups,
    int num_experts,
    bool dense_packed,
    int blocks,
    cudaStream_t stream,
    bool use_pdl,
    Qwen35GroupedMetadataDevice grouped_metadata = {}) {
  if (use_pdl) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(blocks);
    config.blockDim = dim3(kScaleThreads);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id =
        cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        silu_and_mul_mxfp8_grouped_kernel<Source, true>,
        input,
        output,
        scales,
        expert_offsets,
        scale_offsets,
        groups_per_row,
        total_groups,
        num_experts,
        dense_packed,
        grouped_metadata));
  } else {
    silu_and_mul_mxfp8_grouped_kernel<Source, false>
        <<<blocks, kScaleThreads, 0, stream>>>(
            input,
            output,
            scales,
            expert_offsets,
            scale_offsets,
            groups_per_row,
            total_groups,
            num_experts,
            dense_packed,
            grouped_metadata);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
silu_and_mul_mxfp8_grouped_cuda(
    at::Tensor const& input,
    at::Tensor const& expert_offsets,
    std::optional<at::Tensor> scale_offsets_arg) {
  TORCH_CHECK(
      input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
          (input.scalar_type() == at::kHalf ||
           input.scalar_type() == at::kBFloat16),
      "input must be a contiguous CUDA float16/bfloat16 [M,2K] tensor");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.device() == input.device() &&
          expert_offsets.scalar_type() == at::kLong &&
          expert_offsets.is_contiguous() && expert_offsets.dim() == 1 &&
          expert_offsets.numel() >= 2,
      "expert_offsets must be a contiguous CUDA int64 tensor with E+1 entries");

  int64_t const rows = input.size(0);
  int64_t const input_features = input.size(1);
  int64_t const output_features = input_features / 2;
  int64_t const groups_per_row =
      output_features / kMxScaleVectorSize;
  int64_t const num_experts = expert_offsets.numel() - 1;
  TORCH_CHECK(
      rows > 0 && rows <= std::numeric_limits<int>::max(),
      "input rows must be positive and fit int32");
  TORCH_CHECK(
      input_features % 2 == 0 &&
          output_features > 0 &&
          output_features % kMxScaleVectorSize == 0,
      "input last dimension must be twice a positive multiple of 32");
  TORCH_CHECK(
      groups_per_row <= std::numeric_limits<int>::max() &&
          num_experts <= std::numeric_limits<int>::max(),
      "problem dimensions must fit int32");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto byte_options = input.options().dtype(at::kByte);
  auto output = at::empty(
      {rows, output_features}, byte_options);
  int64_t const active_expert_bound = std::min(rows, num_experts);
  int64_t const max_padded_rows =
      rows + active_expert_bound * (kScaleRowsPerAtom - 1);
  int64_t const packed_groups_per_row =
      (groups_per_row + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  auto scales = at::empty(
      {max_padded_rows * packed_groups_per_row}, byte_options);
  at::Tensor scale_offsets;
  if (scale_offsets_arg.has_value()) {
    scale_offsets = *scale_offsets_arg;
    TORCH_CHECK(
        scale_offsets.is_cuda() &&
            scale_offsets.device() == input.device() &&
            scale_offsets.scalar_type() == at::kLong &&
            scale_offsets.is_contiguous() &&
            scale_offsets.dim() == 1 &&
            scale_offsets.numel() == num_experts + 1,
        "scale_offsets must be a contiguous CUDA int64 tensor with E+1 "
        "entries");
  } else {
    scale_offsets = at::empty(
        {num_experts + 1}, expert_offsets.options());
  }

  int const device_index = input.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  C10_CUDA_CHECK(cudaMemsetAsync(
      scales.data_ptr<uint8_t>(), kUe8m0One,
      static_cast<size_t>(scales.numel()), stream.stream()));
  if (!scale_offsets_arg.has_value()) {
    prepare_grouped_scale_offsets<<<1, 1, 0, stream.stream()>>>(
        expert_offsets.data_ptr<int64_t>(),
        scale_offsets.data_ptr<int64_t>(),
        static_cast<int>(num_experts));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  int64_t const total_groups = rows * groups_per_row;
  int64_t const blocks =
      (total_groups + kQuantGroupsPerBlock - 1) /
      kQuantGroupsPerBlock;
  TORCH_CHECK(
      blocks <= std::numeric_limits<int>::max(),
      "SiLU MXFP8 quantization launch grid is too large");
  if (input.scalar_type() == at::kHalf) {
    silu_and_mul_mxfp8_grouped_kernel<at::Half>
        <<<static_cast<int>(blocks), kScaleThreads, 0, stream.stream()>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<uint8_t>(),
            scales.data_ptr<uint8_t>(),
            expert_offsets.data_ptr<int64_t>(),
            scale_offsets.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            total_groups,
            static_cast<int>(num_experts),
            false,
            {});
  } else {
    silu_and_mul_mxfp8_grouped_kernel<at::BFloat16>
        <<<static_cast<int>(blocks), kScaleThreads, 0, stream.stream()>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<uint8_t>(),
            scales.data_ptr<uint8_t>(),
            expert_offsets.data_ptr<int64_t>(),
            scale_offsets.data_ptr<int64_t>(),
            static_cast<int>(groups_per_row),
            total_groups,
            static_cast<int>(num_experts),
            false,
            {});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales, scale_offsets};
}

void silu_and_mul_mxfp8_grouped_out_cuda(
    at::Tensor& output,
    at::Tensor& scales,
    at::Tensor const& input,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    bool use_pdl,
    std::optional<at::Tensor> grouped_output_arg,
    std::optional<at::Tensor> weight_arg,
    std::optional<at::Tensor> weight_scales_arg) {
  TORCH_CHECK(
      input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
          (input.scalar_type() == at::kHalf ||
           input.scalar_type() == at::kBFloat16),
      "input must be contiguous CUDA float16/bfloat16 [M,2K]");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto const device = input.device();
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.device() == device &&
          expert_offsets.scalar_type() == at::kLong &&
          expert_offsets.is_contiguous() &&
          expert_offsets.dim() == 1 &&
          expert_offsets.numel() >= 2,
      "expert_offsets must be contiguous CUDA int64 [E+1]");
  TORCH_CHECK(
      scale_offsets.is_cuda() &&
          scale_offsets.device() == device &&
          scale_offsets.scalar_type() == at::kLong &&
          scale_offsets.is_contiguous() &&
          scale_offsets.sizes() == expert_offsets.sizes(),
      "scale_offsets must be contiguous CUDA int64 [E+1]");

  int64_t const rows = input.size(0);
  int64_t const input_features = input.size(1);
  int64_t const output_features = input_features / 2;
  int64_t const groups_per_row =
      output_features / kMxScaleVectorSize;
  int64_t const num_experts = expert_offsets.numel() - 1;
  int64_t const active_expert_bound =
      std::min(rows, num_experts);
  int64_t const max_padded_rows =
      rows + active_expert_bound * (kScaleRowsPerAtom - 1);
  int64_t const packed_groups_per_row =
      (groups_per_row + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  TORCH_CHECK(
      rows > 0 && rows <= std::numeric_limits<int>::max() &&
          input_features % 2 == 0 &&
          output_features > 0 &&
          output_features % kMxScaleVectorSize == 0 &&
          num_experts <= std::numeric_limits<int>::max(),
      "input and expert dimensions are unsupported");
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          output.scalar_type() == at::kByte &&
          output.is_contiguous() && output.dim() == 2 &&
          output.size(0) == rows &&
          output.size(1) == output_features,
      "output must be contiguous CUDA uint8 [M,K]");
  TORCH_CHECK(
      scales.is_cuda() && scales.device() == device &&
          scales.scalar_type() == at::kByte &&
          scales.is_contiguous() &&
          scales.numel() >=
              max_padded_rows * packed_groups_per_row,
      "scales is too small for grouped SM120 scale packing");

  bool const fused_grouped = grouped_output_arg.has_value();
  TORCH_CHECK(
      fused_grouped == weight_arg.has_value() &&
          fused_grouped == weight_scales_arg.has_value(),
      "grouped_output, weight, and weight_scales must be provided together");
  std::optional<Qwen35GroupedMetadataStorage> grouped_metadata;
  Qwen35GroupedMetadataDevice grouped_metadata_view{};
  if (fused_grouped) {
    auto& grouped_output = *grouped_output_arg;
    auto const& weight = *weight_arg;
    auto const& weight_scales = *weight_scales_arg;
    int64_t const packed_k = weight.size(2);
    int64_t const gemm_k = packed_k * 4 / 3;
    TORCH_CHECK(
        num_experts == qwen35_router::kRoutedExperts &&
            rows >= num_experts &&
            weight.is_cuda() && weight.device() == device &&
            weight.scalar_type() == at::kByte &&
            weight.is_contiguous() && weight.dim() == 3 &&
            weight.size(0) == num_experts &&
            packed_k * 4 == gemm_k * 3 &&
            gemm_k == output_features,
        "fused grouped W2 requires uint8 MXFP6 [256,N,packed_K] ");
    TORCH_CHECK(
        weight_scales.is_cuda() &&
            weight_scales.device() == device &&
            weight_scales.scalar_type() == at::kByte &&
            weight_scales.is_contiguous(),
        "fused grouped W2 weight_scales must be contiguous CUDA uint8");
    TORCH_CHECK(
        grouped_output.is_cuda() &&
            grouped_output.device() == device &&
            grouped_output.scalar_type() == at::kBFloat16 &&
            grouped_output.is_contiguous() &&
            grouped_output.dim() == 2 &&
            grouped_output.size(0) == rows &&
            grouped_output.size(1) == weight.size(1),
        "fused grouped W2 output must be contiguous CUDA bfloat16 [M,N]");
    grouped_metadata.emplace(
        static_cast<int>(num_experts), output);
    grouped_metadata_view = grouped_metadata->device_view(
        output,
        scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
        grouped_output);
  }

  int64_t const total_groups = rows * groups_per_row;
  int64_t const blocks =
      (total_groups + kQuantGroupsPerBlock - 1) /
      kQuantGroupsPerBlock;
  TORCH_CHECK(
      blocks <= std::numeric_limits<int>::max(),
      "SiLU MXFP8 quantization launch grid is too large");
  auto stream =
      c10::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    launch_silu_and_mul_mxfp8_grouped(
        input.data_ptr<at::Half>(),
        output.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(),
        expert_offsets.data_ptr<int64_t>(),
        scale_offsets.data_ptr<int64_t>(),
        static_cast<int>(groups_per_row),
        total_groups,
        static_cast<int>(num_experts),
        false,
        static_cast<int>(blocks),
        stream.stream(),
        use_pdl,
        grouped_metadata_view);
  } else {
    launch_silu_and_mul_mxfp8_grouped(
        input.data_ptr<at::BFloat16>(),
        output.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(),
        expert_offsets.data_ptr<int64_t>(),
        scale_offsets.data_ptr<int64_t>(),
        static_cast<int>(groups_per_row),
        total_groups,
        static_cast<int>(num_experts),
        false,
        static_cast<int>(blocks),
        stream.stream(),
        use_pdl,
        grouped_metadata_view);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (fused_grouped) {
    launch_qwen35_grouped_from_metadata(
        *grouped_output_arg,
        output,
        *grouped_metadata,
        use_pdl);
  }
}

std::tuple<at::Tensor, at::Tensor>
silu_and_mul_mxfp8_logical_cuda(at::Tensor const& input) {
  TORCH_CHECK(
      input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
          (input.scalar_type() == at::kHalf ||
           input.scalar_type() == at::kBFloat16),
      "input must be a contiguous CUDA float16/bfloat16 [M,2K] tensor");

  int64_t const rows = input.size(0);
  int64_t const input_features = input.size(1);
  int64_t const output_features = input_features / 2;
  int64_t const groups_per_row =
      output_features / kMxScaleVectorSize;
  TORCH_CHECK(
      rows > 0 && rows <= std::numeric_limits<int>::max(),
      "input rows must be positive and fit int32");
  TORCH_CHECK(
      input_features % 2 == 0 &&
          output_features > 0 &&
          output_features % kMxScaleVectorSize == 0,
      "input last dimension must be twice a positive multiple of 32");
  TORCH_CHECK(
      groups_per_row <= std::numeric_limits<int>::max(),
      "problem dimensions must fit int32");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto byte_options = input.options().dtype(at::kByte);
  auto output = at::empty(
      {rows, output_features}, byte_options);
  auto scales = at::empty(
      {rows, groups_per_row}, byte_options);

  int const device_index = input.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  int64_t const total_groups = rows * groups_per_row;
  int64_t const blocks =
      (total_groups + kQuantGroupsPerBlock - 1) /
      kQuantGroupsPerBlock;
  TORCH_CHECK(
      blocks <= std::numeric_limits<int>::max(),
      "SiLU MXFP8 quantization launch grid is too large");
  if (input.scalar_type() == at::kHalf) {
    silu_and_mul_mxfp8_grouped_kernel<at::Half>
        <<<static_cast<int>(blocks), kScaleThreads, 0, stream.stream()>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<uint8_t>(),
            scales.data_ptr<uint8_t>(),
            nullptr,
            nullptr,
            static_cast<int>(groups_per_row),
            total_groups,
            0,
            false,
            {});
  } else {
    silu_and_mul_mxfp8_grouped_kernel<at::BFloat16>
        <<<static_cast<int>(blocks), kScaleThreads, 0, stream.stream()>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<uint8_t>(),
            scales.data_ptr<uint8_t>(),
            nullptr,
            nullptr,
            static_cast<int>(groups_per_row),
            total_groups,
            0,
            false,
            {});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}

void silu_and_mul_mxfp8_packed_out_cuda(
    at::Tensor& output,
    at::Tensor& packed_scales,
    at::Tensor const& input) {
  TORCH_CHECK(
      input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
          (input.scalar_type() == at::kHalf ||
           input.scalar_type() == at::kBFloat16),
      "input must be contiguous CUDA float16/bfloat16 [M,2K]");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto const device = input.device();
  int64_t const rows = input.size(0);
  int64_t const input_features = input.size(1);
  int64_t const output_features = input_features / 2;
  int64_t const groups_per_row =
      output_features / kMxScaleVectorSize;
  int64_t const packed_groups_per_row =
      (groups_per_row + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  int64_t const padded_rows =
      (rows + kScaleRowsPerAtom - 1) /
      kScaleRowsPerAtom * kScaleRowsPerAtom;
  TORCH_CHECK(
      rows > 0 && rows <= std::numeric_limits<int>::max() &&
          input_features % 2 == 0 &&
          output_features > 0 &&
          output_features % kMxScaleVectorSize == 0,
      "input dimensions must be positive, fit int32, and K divisible by 32");
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          output.scalar_type() == at::kByte &&
          output.is_contiguous() && output.dim() == 2 &&
          output.size(0) == rows &&
          output.size(1) == output_features,
      "output must be contiguous CUDA uint8 [M,K]");
  TORCH_CHECK(
      packed_scales.is_cuda() &&
          packed_scales.device() == device &&
          packed_scales.scalar_type() == at::kByte &&
          packed_scales.is_contiguous() &&
          packed_scales.numel() >=
              padded_rows * packed_groups_per_row,
      "packed_scales is too small for the SM120 scale layout");

  int64_t const total_groups = rows * groups_per_row;
  int64_t const blocks =
      (total_groups + kQuantGroupsPerBlock - 1) /
      kQuantGroupsPerBlock;
  TORCH_CHECK(
      blocks <= std::numeric_limits<int>::max(),
      "SiLU MXFP8 quantization launch grid is too large");
  auto stream =
      c10::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    silu_and_mul_mxfp8_grouped_kernel<at::Half>
        <<<static_cast<int>(blocks),
           kScaleThreads,
           0,
           stream.stream()>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<uint8_t>(),
            packed_scales.data_ptr<uint8_t>(),
            nullptr,
            nullptr,
            static_cast<int>(groups_per_row),
            total_groups,
            0,
            true,
            {});
  } else {
    silu_and_mul_mxfp8_grouped_kernel<at::BFloat16>
        <<<static_cast<int>(blocks),
           kScaleThreads,
           0,
           stream.stream()>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<uint8_t>(),
            packed_scales.data_ptr<uint8_t>(),
            nullptr,
            nullptr,
            static_cast<int>(groups_per_row),
            total_groups,
            0,
            true,
            {});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class T>
at::Tensor metadata_tensor(int64_t count, at::Tensor const& anchor) {
  static_assert(std::is_trivially_copyable_v<T>);
  return at::empty(
      {count * static_cast<int64_t>(sizeof(T))},
      anchor.options().dtype(at::kByte));
}

template <class Kernel, class Id>
__global__ void prepare_array_w6a8_metadata(
    uint8_t const* activation,
    uint8_t const* logical_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    Id const* topk_ids,
    typename Kernel::ElementD* output,
    uint8_t* route_scales,
    int routes,
    int routed_topk,
    int include_shared,
    int activation_rows,
    int num_experts,
    int n,
    int k,
    typename Kernel::ElementA const** ptr_a,
    typename Kernel::ElementB const** ptr_b,
    typename Kernel::ElementSF const** ptr_sfa,
    typename Kernel::ElementSF const** ptr_sfb,
    typename Kernel::ElementD** ptr_d,
    typename mxfp6_gemm::grouped::ProblemShape::
        UnderlyingProblemShape* problem_shapes,
    typename Kernel::StrideA* stride_a,
    typename Kernel::StrideB* stride_b,
    typename Kernel::StrideC* stride_c,
    typename Kernel::StrideD* stride_d,
    typename Kernel::LayoutSFA* layout_sfa,
    typename Kernel::LayoutSFB* layout_sfb) {
  int const route = blockIdx.x;
  int const routes_per_token = routed_topk + include_shared;
  int const token = route / routes_per_token;
  int const lane = route - token * routes_per_token;
  int const expert =
      include_shared && lane == routed_topk
      ? num_experts - 1
      : static_cast<int>(
            topk_ids[static_cast<int64_t>(token) * routed_topk + lane]);
  if (expert < 0 || expert >= num_experts) {
    return;
  }
  int const source_row =
      activation_rows == routes ? route : token;
  int const k_blocks = k / kMxScaleVectorSize;
  int const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  int64_t const route_scale_offset =
      static_cast<int64_t>(route) *
      kScaleRowsPerAtom * packed_k_blocks;

  if (threadIdx.x == 0) {
    ptr_a[route] =
        reinterpret_cast<typename Kernel::ElementA const*>(
            weight + static_cast<int64_t>(expert) * n * k * 3 / 4);
    ptr_b[route] =
        reinterpret_cast<typename Kernel::ElementB const*>(
            activation + static_cast<int64_t>(source_row) * k);
    ptr_sfa[route] =
        reinterpret_cast<typename Kernel::ElementSF const*>(
            weight_scales +
            static_cast<int64_t>(expert) * n * packed_k_blocks);
    ptr_sfb[route] =
        reinterpret_cast<typename Kernel::ElementSF const*>(
            route_scales + route_scale_offset);
    ptr_d[route] = output + static_cast<int64_t>(route) * n;
    auto const problem = cute::make_shape(n, 1, k, 1);
    auto const scale_problem =
        cute::make_shape(n, 1, max(k, 128), 1);
    problem_shapes[route] = cute::make_shape(n, 1, k);
    stride_a[route] = cutlass::make_cute_packed_stride(
        typename Kernel::StrideA{}, cute::make_shape(n, k, 1));
    stride_b[route] = cutlass::make_cute_packed_stride(
        typename Kernel::StrideB{}, cute::make_shape(1, k, 1));
    stride_c[route] = cutlass::make_cute_packed_stride(
        typename Kernel::StrideC{}, cute::make_shape(n, 1, 1));
    stride_d[route] = cutlass::make_cute_packed_stride(
        typename Kernel::StrideD{}, cute::make_shape(n, 1, 1));
    layout_sfa[route] =
        Kernel::BlockScaledConfig::tile_atom_to_shape_SFA(
            scale_problem);
    layout_sfb[route] =
        Kernel::BlockScaledConfig::tile_atom_to_shape_SFB(
            scale_problem);
  }
  auto const* source_scales =
      logical_scales + static_cast<int64_t>(source_row) * k_blocks;
  for (int k_block = threadIdx.x;
       k_block < k_blocks;
       k_block += blockDim.x) {
    route_scales[
        route_scale_offset +
        grouped_scale_offset(0, k_block, k_blocks)] =
        source_scales[k_block];
  }
}

template <class Kernel, class Id>
__global__ void prepare_uniform_array_w6a8_metadata(
    uint8_t const* activation,
    uint8_t const* logical_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    Id const* topk_ids,
    typename Kernel::ElementD* output,
    uint8_t* route_scales,
    int routed_topk,
    int include_shared,
    int activation_rows,
    int num_experts,
    int n,
    int k,
    typename Kernel::ElementA const** ptr_a,
    typename Kernel::ElementB const** ptr_b,
    typename Kernel::ElementSF const** ptr_sfa,
    typename Kernel::ElementSF const** ptr_sfb,
    typename Kernel::ElementD** ptr_d) {
  int const route = blockIdx.x;
  int const routes_per_token = routed_topk + include_shared;
  int const token = route / routes_per_token;
  int const lane = route - token * routes_per_token;
  int const expert =
      include_shared && lane == routed_topk
      ? num_experts - 1
      : static_cast<int>(
            topk_ids[static_cast<int64_t>(token) * routed_topk + lane]);
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  int const source_row =
      activation_rows == gridDim.x ? route : token;
  int const k_blocks = k / kMxScaleVectorSize;
  int const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  int64_t const route_scale_offset =
      static_cast<int64_t>(route) *
      kScaleRowsPerAtom * packed_k_blocks;

  if (threadIdx.x == 0) {
    ptr_a[route] =
        reinterpret_cast<typename Kernel::ElementA const*>(
            weight + static_cast<int64_t>(expert) * n * k * 3 / 4);
    ptr_b[route] =
        reinterpret_cast<typename Kernel::ElementB const*>(
            activation + static_cast<int64_t>(source_row) * k);
    ptr_sfa[route] =
        reinterpret_cast<typename Kernel::ElementSF const*>(
            weight_scales +
            static_cast<int64_t>(expert) * n * packed_k_blocks);
    ptr_sfb[route] =
        reinterpret_cast<typename Kernel::ElementSF const*>(
            route_scales + route_scale_offset);
    ptr_d[route] = output + static_cast<int64_t>(route) * n;
  }

  auto const* source_scales =
      logical_scales + static_cast<int64_t>(source_row) * k_blocks;
  for (int k_block = threadIdx.x;
       k_block < k_blocks;
       k_block += blockDim.x) {
    route_scales[
        route_scale_offset +
        grouped_scale_offset(0, k_block, k_blocks)] =
        source_scales[k_block];
  }
}

template <class Kernel>
void launch_uniform_array_w6a8(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    int n,
    int k,
    bool include_shared) {
  using ProblemShape = mxfp6_gemm::grouped::ArrayProblemShape;
  using GemmKernel = typename Kernel::GemmKernel;
  using Gemm = typename Kernel::Gemm;

  int const tokens = static_cast<int>(topk_ids.size(0));
  int const routed_topk = static_cast<int>(topk_ids.size(1));
  int const routes =
      tokens * (routed_topk + static_cast<int>(include_shared));
  int const activation_rows = static_cast<int>(activation.size(0));
  int const num_experts = static_cast<int>(weight.size(0));
  int const k_blocks = k / kMxScaleVectorSize;
  int const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;

  auto ptr_a = metadata_tensor<typename Kernel::ElementA const*>(
      routes, activation);
  auto ptr_b = metadata_tensor<typename Kernel::ElementB const*>(
      routes, activation);
  auto ptr_sfa = metadata_tensor<typename Kernel::ElementSF const*>(
      routes, activation);
  auto ptr_sfb = metadata_tensor<typename Kernel::ElementSF const*>(
      routes, activation);
  auto ptr_d = metadata_tensor<typename Kernel::ElementD*>(
      routes, activation);
  auto route_scales = at::empty(
      {static_cast<int64_t>(routes) *
       kScaleRowsPerAtom * packed_k_blocks},
      activation.options().dtype(at::kByte));

  int const device_index = activation.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  C10_CUDA_CHECK(cudaMemsetAsync(
      route_scales.data_ptr<uint8_t>(), kUe8m0One,
      static_cast<size_t>(route_scales.numel()), stream.stream()));
  constexpr int threads = 128;
  if (topk_ids.scalar_type() == at::kInt) {
    prepare_uniform_array_w6a8_metadata<Kernel, int32_t>
        <<<routes, threads, 0, stream.stream()>>>(
            static_cast<uint8_t const*>(activation.data_ptr()),
            logical_scales.data_ptr<uint8_t>(),
            weight.data_ptr<uint8_t>(),
            weight_scales.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int32_t>(),
            reinterpret_cast<typename Kernel::ElementD*>(
                output.data_ptr()),
            route_scales.data_ptr<uint8_t>(),
            routed_topk,
            static_cast<int>(include_shared),
            activation_rows,
            num_experts,
            n,
            k,
            reinterpret_cast<typename Kernel::ElementA const**>(
                ptr_a.data_ptr()),
            reinterpret_cast<typename Kernel::ElementB const**>(
                ptr_b.data_ptr()),
            reinterpret_cast<typename Kernel::ElementSF const**>(
                ptr_sfa.data_ptr()),
            reinterpret_cast<typename Kernel::ElementSF const**>(
                ptr_sfb.data_ptr()),
            reinterpret_cast<typename Kernel::ElementD**>(
                ptr_d.data_ptr()));
  } else {
    prepare_uniform_array_w6a8_metadata<Kernel, int64_t>
        <<<routes, threads, 0, stream.stream()>>>(
            static_cast<uint8_t const*>(activation.data_ptr()),
            logical_scales.data_ptr<uint8_t>(),
            weight.data_ptr<uint8_t>(),
            weight_scales.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int64_t>(),
            reinterpret_cast<typename Kernel::ElementD*>(
                output.data_ptr()),
            route_scales.data_ptr<uint8_t>(),
            routed_topk,
            static_cast<int>(include_shared),
            activation_rows,
            num_experts,
            n,
            k,
            reinterpret_cast<typename Kernel::ElementA const**>(
                ptr_a.data_ptr()),
            reinterpret_cast<typename Kernel::ElementB const**>(
                ptr_b.data_ptr()),
            reinterpret_cast<typename Kernel::ElementSF const**>(
                ptr_sfa.data_ptr()),
            reinterpret_cast<typename Kernel::ElementSF const**>(
                ptr_sfb.data_ptr()),
            reinterpret_cast<typename Kernel::ElementD**>(
                ptr_d.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto const problem_shape =
      cute::make_shape(n, 1, k, routes);
  ProblemShape problem{problem_shape};
  auto const stride_a = cutlass::make_cute_packed_stride(
      typename Kernel::StrideA{},
      cute::make_shape(n, k, 1));
  auto const stride_b = cutlass::make_cute_packed_stride(
      typename Kernel::StrideB{},
      cute::make_shape(1, k, 1));
  auto const stride_c = cutlass::make_cute_packed_stride(
      typename Kernel::StrideC{},
      cute::make_shape(n, 1, 1));
  auto const stride_d = cutlass::make_cute_packed_stride(
      typename Kernel::StrideD{},
      cute::make_shape(n, 1, 1));
  auto const scale_problem =
      cute::make_shape(n, 1, max(k, 128), 1);
  auto const layout_sfa =
      Kernel::BlockScaledConfig::tile_atom_to_shape_SFA(scale_problem);
  auto const layout_sfb =
      Kernel::BlockScaledConfig::tile_atom_to_shape_SFB(scale_problem);

  typename GemmKernel::MainloopArguments mainloop{
      reinterpret_cast<typename Kernel::ElementA const**>(
          ptr_a.data_ptr()),
      stride_a,
      reinterpret_cast<typename Kernel::ElementB const**>(
          ptr_b.data_ptr()),
      stride_b,
      reinterpret_cast<typename Kernel::ElementSF const**>(
          ptr_sfa.data_ptr()),
      layout_sfa,
      reinterpret_cast<typename Kernel::ElementSF const**>(
          ptr_sfb.data_ptr()),
      layout_sfb};
  typename GemmKernel::EpilogueArguments epilogue{
      {1.0f, 0.0f},
      nullptr,
      stride_c,
      reinterpret_cast<typename Kernel::ElementD**>(ptr_d.data_ptr()),
      stride_d};
  cutlass::KernelHardwareInfo hw_info{
      device_index,
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          device_index)};
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kArray,
      problem,
      mainloop,
      epilogue,
      hw_info};

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "uniform-array MXFP6 can_implement failed: ",
      cutlassGetStatusString(status));
  size_t const workspace_bytes = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_bytes)},
        activation.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream.stream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "uniform-array MXFP6 initialize failed: ",
      cutlassGetStatusString(status));
  status = gemm.run(stream.stream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "uniform-array MXFP6 launch failed: ",
      cutlassGetStatusString(status));
}

template <class Kernel>
void launch_array_w6a8(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    int n,
    int k,
    bool include_shared) {
  using ProblemShape = mxfp6_gemm::grouped::ProblemShape;
  using GemmKernel = typename Kernel::GemmKernel;
  using Gemm = typename Kernel::Gemm;

  int const tokens = static_cast<int>(topk_ids.size(0));
  int const routed_topk = static_cast<int>(topk_ids.size(1));
  int const routes =
      tokens * (routed_topk + static_cast<int>(include_shared));
  int const activation_rows = static_cast<int>(activation.size(0));
  int const num_experts = static_cast<int>(weight.size(0));
  int const k_blocks = k / kMxScaleVectorSize;
  int const packed_k_blocks =
      (k_blocks + kScaleGroupsPerAtom - 1) /
      kScaleGroupsPerAtom * kScaleGroupsPerAtom;
  auto ptr_a = metadata_tensor<typename Kernel::ElementA const*>(
      routes, activation);
  auto ptr_b = metadata_tensor<typename Kernel::ElementB const*>(
      routes, activation);
  auto ptr_sfa = metadata_tensor<typename Kernel::ElementSF const*>(
      routes, activation);
  auto ptr_sfb = metadata_tensor<typename Kernel::ElementSF const*>(
      routes, activation);
  auto ptr_d = metadata_tensor<typename Kernel::ElementD*>(
      routes, activation);
  auto problem_shapes =
      metadata_tensor<typename ProblemShape::UnderlyingProblemShape>(
          routes, activation);
  auto stride_a =
      metadata_tensor<typename Kernel::StrideA>(routes, activation);
  auto stride_b =
      metadata_tensor<typename Kernel::StrideB>(routes, activation);
  auto stride_c =
      metadata_tensor<typename Kernel::StrideC>(routes, activation);
  auto stride_d =
      metadata_tensor<typename Kernel::StrideD>(routes, activation);
  auto layout_sfa =
      metadata_tensor<typename Kernel::LayoutSFA>(routes, activation);
  auto layout_sfb =
      metadata_tensor<typename Kernel::LayoutSFB>(routes, activation);
  auto route_scales = at::empty(
      {static_cast<int64_t>(routes) *
       kScaleRowsPerAtom * packed_k_blocks},
      activation.options().dtype(at::kByte));

  int const device_index = activation.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  C10_CUDA_CHECK(cudaMemsetAsync(
      route_scales.data_ptr<uint8_t>(), kUe8m0One,
      static_cast<size_t>(route_scales.numel()), stream.stream()));
  constexpr int threads = 128;
  if (topk_ids.scalar_type() == at::kInt) {
    prepare_array_w6a8_metadata<Kernel, int32_t>
        <<<routes, threads, 0, stream.stream()>>>(
            static_cast<uint8_t const*>(activation.data_ptr()),
            logical_scales.data_ptr<uint8_t>(),
            weight.data_ptr<uint8_t>(),
            weight_scales.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int32_t>(),
            reinterpret_cast<typename Kernel::ElementD*>(
                output.data_ptr()),
            route_scales.data_ptr<uint8_t>(),
            routes,
            routed_topk,
            static_cast<int>(include_shared),
            activation_rows,
            num_experts,
            n,
            k,
            reinterpret_cast<typename Kernel::ElementA const**>(
                ptr_a.data_ptr()),
            reinterpret_cast<typename Kernel::ElementB const**>(
                ptr_b.data_ptr()),
            reinterpret_cast<typename Kernel::ElementSF const**>(
                ptr_sfa.data_ptr()),
            reinterpret_cast<typename Kernel::ElementSF const**>(
                ptr_sfb.data_ptr()),
            reinterpret_cast<typename Kernel::ElementD**>(
                ptr_d.data_ptr()),
            reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
                problem_shapes.data_ptr()),
            reinterpret_cast<typename Kernel::StrideA*>(
                stride_a.data_ptr()),
            reinterpret_cast<typename Kernel::StrideB*>(
                stride_b.data_ptr()),
            reinterpret_cast<typename Kernel::StrideC*>(
                stride_c.data_ptr()),
            reinterpret_cast<typename Kernel::StrideD*>(
                stride_d.data_ptr()),
            reinterpret_cast<typename Kernel::LayoutSFA*>(
                layout_sfa.data_ptr()),
            reinterpret_cast<typename Kernel::LayoutSFB*>(
                layout_sfb.data_ptr()));
  } else {
    prepare_array_w6a8_metadata<Kernel, int64_t>
        <<<routes, threads, 0, stream.stream()>>>(
            static_cast<uint8_t const*>(activation.data_ptr()),
            logical_scales.data_ptr<uint8_t>(),
            weight.data_ptr<uint8_t>(),
            weight_scales.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int64_t>(),
            reinterpret_cast<typename Kernel::ElementD*>(
                output.data_ptr()),
            route_scales.data_ptr<uint8_t>(),
            routes,
            routed_topk,
            static_cast<int>(include_shared),
            activation_rows,
            num_experts,
            n,
            k,
            reinterpret_cast<typename Kernel::ElementA const**>(
                ptr_a.data_ptr()),
            reinterpret_cast<typename Kernel::ElementB const**>(
                ptr_b.data_ptr()),
            reinterpret_cast<typename Kernel::ElementSF const**>(
                ptr_sfa.data_ptr()),
            reinterpret_cast<typename Kernel::ElementSF const**>(
                ptr_sfb.data_ptr()),
            reinterpret_cast<typename Kernel::ElementD**>(
                ptr_d.data_ptr()),
            reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
                problem_shapes.data_ptr()),
            reinterpret_cast<typename Kernel::StrideA*>(
                stride_a.data_ptr()),
            reinterpret_cast<typename Kernel::StrideB*>(
                stride_b.data_ptr()),
            reinterpret_cast<typename Kernel::StrideC*>(
                stride_c.data_ptr()),
            reinterpret_cast<typename Kernel::StrideD*>(
                stride_d.data_ptr()),
            reinterpret_cast<typename Kernel::LayoutSFA*>(
                layout_sfa.data_ptr()),
            reinterpret_cast<typename Kernel::LayoutSFB*>(
                layout_sfb.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ProblemShape problem{
      routes,
      reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
          problem_shapes.data_ptr()),
      nullptr};
  typename GemmKernel::MainloopArguments mainloop{
      reinterpret_cast<typename Kernel::ElementA const**>(
          ptr_a.data_ptr()),
      reinterpret_cast<typename Kernel::StrideA*>(stride_a.data_ptr()),
      reinterpret_cast<typename Kernel::ElementB const**>(
          ptr_b.data_ptr()),
      reinterpret_cast<typename Kernel::StrideB*>(stride_b.data_ptr()),
      reinterpret_cast<typename Kernel::ElementSF const**>(
          ptr_sfa.data_ptr()),
      reinterpret_cast<typename Kernel::LayoutSFA*>(
          layout_sfa.data_ptr()),
      reinterpret_cast<typename Kernel::ElementSF const**>(
          ptr_sfb.data_ptr()),
      reinterpret_cast<typename Kernel::LayoutSFB*>(
          layout_sfb.data_ptr())};
  typename GemmKernel::EpilogueArguments epilogue{
      {1.0f, 0.0f},
      nullptr,
      reinterpret_cast<typename Kernel::StrideC*>(stride_c.data_ptr()),
      reinterpret_cast<typename Kernel::ElementD**>(ptr_d.data_ptr()),
      reinterpret_cast<typename Kernel::StrideD*>(stride_d.data_ptr())};
  cutlass::KernelHardwareInfo hw_info{
      device_index,
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          device_index)};
  typename GemmKernel::TileSchedulerArguments scheduler;
  scheduler.max_swizzle_size = 1;
  scheduler.uniform_groups = true;
  scheduler.uniform_tiles_per_group =
      static_cast<uint32_t>(
          cute::ceil_div(n, cute::size<0>(typename Kernel::TileShape{})));
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      problem,
      mainloop,
      epilogue,
      hw_info,
      scheduler};

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "array MXFP6 can_implement failed: ",
      cutlassGetStatusString(status));
  size_t const workspace_bytes = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_bytes)},
        activation.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream.stream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "array MXFP6 initialize failed: ",
      cutlassGetStatusString(status));
  status = gemm.run(stream.stream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "array MXFP6 launch failed: ",
      cutlassGetStatusString(status));
}

template <class Kernel, bool UsePdl = false>
__global__ void prepare_grouped_w6a8_metadata(
    uint8_t const* activation,
    uint8_t const* activation_scales,
    uint8_t const* weight,
    uint8_t const* weight_scales,
    int64_t const* expert_offsets,
    int64_t const* scale_offsets,
    typename Kernel::ElementD* output,
    int num_experts,
    int group_count,
    int n,
    int k,
    typename mxfp6_gemm::grouped::ProblemShape::UnderlyingProblemShape*
        problem_shapes,
    typename Kernel::ElementA const** ptr_a,
    typename Kernel::ElementB const** ptr_b,
    typename Kernel::ElementSF const** ptr_sfa,
    typename Kernel::ElementSF const** ptr_sfb,
    typename Kernel::ElementD** ptr_d,
    typename Kernel::StrideA* stride_a,
    typename Kernel::StrideB* stride_b,
    typename Kernel::StrideC* stride_c,
    typename Kernel::StrideD* stride_d,
    typename Kernel::LayoutSFA* layout_sfa,
    typename Kernel::LayoutSFB* layout_sfb) {
  int const group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= group_count) {
    return;
  }

  int expert = group;
  bool found = true;
  if (group_count < num_experts) {
    found = false;
    int active_index = 0;
    for (int candidate = 0; candidate < num_experts; ++candidate) {
      if (expert_offsets[candidate + 1] > expert_offsets[candidate]) {
        if (active_index == group) {
          expert = candidate;
          found = true;
          break;
        }
        ++active_index;
      }
    }
    if (!found) {
      expert = 0;
    }
  }
  int64_t const row_start = expert_offsets[expert];
  int64_t const row_end = expert_offsets[expert + 1];
  int const rows =
      found ? static_cast<int>(row_end - row_start) : 0;
  auto problem = cute::make_shape(n, rows, k, 1);
  auto scale_problem =
      cute::make_shape(n, rows, max(k, 128), 1);

  problem_shapes[group] = cute::make_shape(n, rows, k);
  ptr_a[group] = reinterpret_cast<typename Kernel::ElementA const*>(
      weight + static_cast<int64_t>(expert) * n * k * 3 / 4);
  ptr_b[group] = reinterpret_cast<typename Kernel::ElementB const*>(
      activation + row_start * k);
  ptr_sfa[group] = reinterpret_cast<typename Kernel::ElementSF const*>(
      weight_scales +
      static_cast<int64_t>(expert) * n *
      ((k / 32 + kScaleGroupsPerAtom - 1) /
       kScaleGroupsPerAtom * kScaleGroupsPerAtom));
  ptr_sfb[group] = reinterpret_cast<typename Kernel::ElementSF const*>(
      activation_scales +
      scale_offsets[expert] *
      ((k / 32 + kScaleGroupsPerAtom - 1) /
       kScaleGroupsPerAtom * kScaleGroupsPerAtom));
  ptr_d[group] = output + row_start * n;

  stride_a[group] = cutlass::make_cute_packed_stride(
      typename Kernel::StrideA{}, cute::make_shape(n, k, 1));
  stride_b[group] = cutlass::make_cute_packed_stride(
      typename Kernel::StrideB{}, cute::make_shape(rows, k, 1));
  stride_c[group] = cutlass::make_cute_packed_stride(
      typename Kernel::StrideC{}, cute::make_shape(n, rows, 1));
  stride_d[group] = cutlass::make_cute_packed_stride(
      typename Kernel::StrideD{}, cute::make_shape(n, rows, 1));
  layout_sfa[group] =
      Kernel::BlockScaledConfig::tile_atom_to_shape_SFA(
          scale_problem);
  layout_sfb[group] =
      Kernel::BlockScaledConfig::tile_atom_to_shape_SFB(
          scale_problem);
  if constexpr (UsePdl) {
    if (threadIdx.x == 0) {
      cutlass::arch::wait_on_dependent_grids();
      cutlass::arch::launch_dependent_grids();
    }
  }
}

template <class Kernel>
void launch_grouped_w6a8(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& activation_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    int n,
    int k,
    bool use_pdl) {
  using ProblemShape = mxfp6_gemm::grouped::ProblemShape;
  using GemmKernel = typename Kernel::GemmKernel;
  using Gemm = typename Kernel::Gemm;

  int const num_experts = static_cast<int>(weight.size(0));
  int const group_count = static_cast<int>(
      std::min<int64_t>(num_experts, activation.size(0)));
  auto problem_shapes =
      metadata_tensor<typename ProblemShape::UnderlyingProblemShape>(
          group_count, activation);
  auto ptr_a = metadata_tensor<typename Kernel::ElementA const*>(
      group_count, activation);
  auto ptr_b = metadata_tensor<typename Kernel::ElementB const*>(
      group_count, activation);
  auto ptr_sfa = metadata_tensor<typename Kernel::ElementSF const*>(
      group_count, activation);
  auto ptr_sfb = metadata_tensor<typename Kernel::ElementSF const*>(
      group_count, activation);
  auto ptr_d = metadata_tensor<typename Kernel::ElementD*>(
      group_count, activation);
  auto stride_a =
      metadata_tensor<typename Kernel::StrideA>(group_count, activation);
  auto stride_b =
      metadata_tensor<typename Kernel::StrideB>(group_count, activation);
  auto stride_c =
      metadata_tensor<typename Kernel::StrideC>(group_count, activation);
  auto stride_d =
      metadata_tensor<typename Kernel::StrideD>(group_count, activation);
  auto layout_sfa =
      metadata_tensor<typename Kernel::LayoutSFA>(group_count, activation);
  auto layout_sfb =
      metadata_tensor<typename Kernel::LayoutSFB>(group_count, activation);

  int const device_index = activation.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  constexpr int threads = 128;
  int const metadata_blocks =
      (group_count + threads - 1) / threads;
  auto const* activation_ptr =
      static_cast<uint8_t const*>(activation.data_ptr());
  auto const* activation_scales_ptr =
      activation_scales.data_ptr<uint8_t>();
  auto const* weight_ptr = weight.data_ptr<uint8_t>();
  auto const* weight_scales_ptr =
      weight_scales.data_ptr<uint8_t>();
  auto const* expert_offsets_ptr =
      expert_offsets.data_ptr<int64_t>();
  auto const* scale_offsets_ptr =
      scale_offsets.data_ptr<int64_t>();
  auto* output_ptr =
      reinterpret_cast<typename Kernel::ElementD*>(
          output.data_ptr());
  auto* problem_shapes_ptr =
      reinterpret_cast<
          typename ProblemShape::UnderlyingProblemShape*>(
          problem_shapes.data_ptr());
  auto* ptr_a_ptr =
      reinterpret_cast<typename Kernel::ElementA const**>(
          ptr_a.data_ptr());
  auto* ptr_b_ptr =
      reinterpret_cast<typename Kernel::ElementB const**>(
          ptr_b.data_ptr());
  auto* ptr_sfa_ptr =
      reinterpret_cast<typename Kernel::ElementSF const**>(
          ptr_sfa.data_ptr());
  auto* ptr_sfb_ptr =
      reinterpret_cast<typename Kernel::ElementSF const**>(
          ptr_sfb.data_ptr());
  auto* ptr_d_ptr =
      reinterpret_cast<typename Kernel::ElementD**>(
          ptr_d.data_ptr());
  auto* stride_a_ptr =
      reinterpret_cast<typename Kernel::StrideA*>(
          stride_a.data_ptr());
  auto* stride_b_ptr =
      reinterpret_cast<typename Kernel::StrideB*>(
          stride_b.data_ptr());
  auto* stride_c_ptr =
      reinterpret_cast<typename Kernel::StrideC*>(
          stride_c.data_ptr());
  auto* stride_d_ptr =
      reinterpret_cast<typename Kernel::StrideD*>(
          stride_d.data_ptr());
  auto* layout_sfa_ptr =
      reinterpret_cast<typename Kernel::LayoutSFA*>(
          layout_sfa.data_ptr());
  auto* layout_sfb_ptr =
      reinterpret_cast<typename Kernel::LayoutSFB*>(
          layout_sfb.data_ptr());
  if (use_pdl) {
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(metadata_blocks);
    config.blockDim = dim3(threads);
    config.stream = stream.stream();
    cudaLaunchAttribute attribute{};
    attribute.id =
        cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        prepare_grouped_w6a8_metadata<Kernel, true>,
        activation_ptr,
        activation_scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        expert_offsets_ptr,
        scale_offsets_ptr,
        output_ptr,
        num_experts,
        group_count,
        n,
        k,
        problem_shapes_ptr,
        ptr_a_ptr,
        ptr_b_ptr,
        ptr_sfa_ptr,
        ptr_sfb_ptr,
        ptr_d_ptr,
        stride_a_ptr,
        stride_b_ptr,
        stride_c_ptr,
        stride_d_ptr,
        layout_sfa_ptr,
        layout_sfb_ptr));
  } else {
    prepare_grouped_w6a8_metadata<Kernel, false>
        <<<metadata_blocks, threads, 0, stream.stream()>>>(
            activation_ptr,
            activation_scales_ptr,
            weight_ptr,
            weight_scales_ptr,
            expert_offsets_ptr,
            scale_offsets_ptr,
            output_ptr,
            num_experts,
            group_count,
            n,
            k,
            problem_shapes_ptr,
            ptr_a_ptr,
            ptr_b_ptr,
            ptr_sfa_ptr,
            ptr_sfb_ptr,
            ptr_d_ptr,
            stride_a_ptr,
            stride_b_ptr,
            stride_c_ptr,
            stride_d_ptr,
            layout_sfa_ptr,
            layout_sfb_ptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ProblemShape problem{
      group_count,
      reinterpret_cast<typename ProblemShape::UnderlyingProblemShape*>(
          problem_shapes.data_ptr()),
      nullptr};
  typename GemmKernel::MainloopArguments mainloop{
      reinterpret_cast<typename Kernel::ElementA const**>(ptr_a.data_ptr()),
      reinterpret_cast<typename Kernel::StrideA*>(stride_a.data_ptr()),
      reinterpret_cast<typename Kernel::ElementB const**>(ptr_b.data_ptr()),
      reinterpret_cast<typename Kernel::StrideB*>(stride_b.data_ptr()),
      reinterpret_cast<typename Kernel::ElementSF const**>(
          ptr_sfa.data_ptr()),
      reinterpret_cast<typename Kernel::LayoutSFA*>(layout_sfa.data_ptr()),
      reinterpret_cast<typename Kernel::ElementSF const**>(
          ptr_sfb.data_ptr()),
      reinterpret_cast<typename Kernel::LayoutSFB*>(layout_sfb.data_ptr()),
      static_cast<int32_t>(n),
      128,
      static_cast<int32_t>(k),
      grouped_fixed_tensormaps() && activation.size(0) <= 1024};

  typename GemmKernel::EpilogueArguments epilogue{
      {1.0f, 0.0f},
      nullptr,
      reinterpret_cast<typename Kernel::StrideC*>(stride_c.data_ptr()),
      reinterpret_cast<typename Kernel::ElementD**>(ptr_d.data_ptr()),
      reinterpret_cast<typename Kernel::StrideD*>(stride_d.data_ptr())};

  int const physical_sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          device_index);
  constexpr bool kDualCtaPersistent =
      GemmKernel::MaxThreadsPerBlock <= 256;
  cutlass::KernelHardwareInfo hw_info{
      device_index,
      physical_sm_count * (kDualCtaPersistent ? 2 : 1)};
  typename GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = grouped_raster_along_n()
      ? GemmKernel::TileScheduler::RasterOrderOptions::AlongN
      : GemmKernel::TileScheduler::RasterOrderOptions::AlongM;
  scheduler.max_swizzle_size = grouped_max_swizzle();
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      problem,
      mainloop,
      epilogue,
      hw_info,
      scheduler};

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "grouped MXFP6 can_implement failed: ",
      cutlassGetStatusString(status));
  size_t const workspace_bytes = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_bytes)},
        activation.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream.stream());
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "grouped MXFP6 initialize failed: ",
      cutlassGetStatusString(status));
  status = gemm.run(stream.stream(), nullptr, use_pdl);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "grouped MXFP6 launch failed: ",
      cutlassGetStatusString(status));
}

void check_cuda_byte_tensor(
    at::Tensor const& tensor,
    char const* name,
    at::Device const& device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == at::kByte, name, " must be uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void array_gemm_w6a8_out_cuda(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    bool include_shared) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  TORCH_CHECK(false, "mxfp6_torch must be compiled for sm_120a");
#else
  TORCH_CHECK(activation.is_cuda(), "activation must be a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(activation.device());
  auto const device = activation.device();
  TORCH_CHECK(
      activation.scalar_type() == at::kFloat8_e4m3fn ||
          activation.scalar_type() == at::kByte,
      "activation must be float8_e4m3fn or uint8");
  TORCH_CHECK(
      activation.dim() == 2 && activation.is_contiguous(),
      "activation must be contiguous [M,K]");
  check_cuda_byte_tensor(logical_scales, "logical_scales", device);
  check_cuda_byte_tensor(weight, "weight", device);
  check_cuda_byte_tensor(weight_scales, "weight_scales", device);
  TORCH_CHECK(
      topk_ids.is_cuda() && topk_ids.device() == device &&
          (topk_ids.scalar_type() == at::kInt ||
           topk_ids.scalar_type() == at::kLong) &&
          topk_ids.is_contiguous() && topk_ids.dim() == 2,
      "topk_ids must be contiguous CUDA int32/int64 [tokens,topk]");
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          output.is_contiguous() && output.dim() == 2 &&
          (output.scalar_type() == at::kHalf ||
           output.scalar_type() == at::kBFloat16),
      "output must be contiguous CUDA float16/bfloat16 [routes,N]");
  TORCH_CHECK(weight.dim() == 3, "weight must have shape [E,N,packed_K]");

  int64_t const tokens = topk_ids.size(0);
  int64_t const topk = topk_ids.size(1);
  int64_t const routes =
      tokens * (topk + static_cast<int64_t>(include_shared));
  int64_t const n = weight.size(1);
  int64_t const packed_k = weight.size(2);
  int64_t const k = packed_k * 4 / 3;
  TORCH_CHECK(
      tokens > 0 && topk > 0 &&
          routes <= std::numeric_limits<int>::max(),
      "routed rows must be positive and fit int32");
  TORCH_CHECK(
      !include_shared || weight.size(0) >= 2,
      "include_shared requires one trailing shared expert");
  TORCH_CHECK(
      n > 0 && n % 128 == 0 &&
          k > 0 && k % kMxScaleVectorSize == 0 &&
          n <= std::numeric_limits<int>::max() &&
          k <= std::numeric_limits<int>::max(),
      "N must be a multiple of 128 and K a multiple of 32");
  TORCH_CHECK(
      packed_k * 4 == k * 3 &&
          activation.size(1) == k,
      "activation K does not match packed weight K");
  TORCH_CHECK(
      activation.size(0) == tokens ||
          activation.size(0) == routes,
      "activation rows must equal tokens or all routed/shared rows");
  TORCH_CHECK(
      logical_scales.dim() == 2 &&
          logical_scales.size(0) == activation.size(0) &&
          logical_scales.size(1) == k / kMxScaleVectorSize,
      "logical_scales must have shape [activation_rows,K/32]");
  TORCH_CHECK(
      output.size(0) == routes && output.size(1) == n,
      "output must contain all routed/shared rows");
  TORCH_CHECK(
      weight_scales.numel() >=
          weight.size(0) * n *
          ((k / 32 + kScaleGroupsPerAtom - 1) /
           kScaleGroupsPerAtom * kScaleGroupsPerAtom),
      "weight_scales is too small");

  int const device_index = activation.get_device();
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(device_index);
  TORCH_CHECK(
      properties.major == 12 && properties.minor == 0,
      "array MXFP6 requires SM120");

  if (k % direct_moe::kTileK == 0) {
    auto stream = c10::cuda::getCurrentCUDAStream(device_index);
    if (output.scalar_type() == at::kBFloat16) {
      if (topk_ids.scalar_type() == at::kInt) {
        direct_moe::launch<int32_t, cutlass::bfloat16_t>(
            output, activation, logical_scales, weight, weight_scales,
            topk_ids, include_shared, stream.stream());
      } else {
        direct_moe::launch<int64_t, cutlass::bfloat16_t>(
            output, activation, logical_scales, weight, weight_scales,
            topk_ids, include_shared, stream.stream());
      }
    } else if (topk_ids.scalar_type() == at::kInt) {
      direct_moe::launch<int32_t, cutlass::half_t>(
          output, activation, logical_scales, weight, weight_scales,
          topk_ids, include_shared, stream.stream());
    } else {
      direct_moe::launch<int64_t, cutlass::half_t>(
          output, activation, logical_scales, weight, weight_scales,
          topk_ids, include_shared, stream.stream());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  if (output.scalar_type() == at::kBFloat16) {
    if (k == 64) {
      using Kernel =
          typename mxfp6_gemm::grouped::Kernel128x8x64::
              template RebindOutput<cutlass::bfloat16_t>;
      launch_array_w6a8<Kernel>(
          output, activation, logical_scales, weight, weight_scales,
          topk_ids, static_cast<int>(n), static_cast<int>(k),
          include_shared);
    } else {
      using Kernel =
          typename mxfp6_gemm::grouped::Kernel128x8x128::
              template RebindOutput<cutlass::bfloat16_t>;
      launch_array_w6a8<Kernel>(
          output, activation, logical_scales, weight, weight_scales,
          topk_ids, static_cast<int>(n), static_cast<int>(k),
          include_shared);
    }
  } else {
    if (k == 64) {
      using Kernel =
          typename mxfp6_gemm::grouped::Kernel128x8x64::
              template RebindOutput<cutlass::half_t>;
      launch_array_w6a8<Kernel>(
          output, activation, logical_scales, weight, weight_scales,
          topk_ids, static_cast<int>(n), static_cast<int>(k),
          include_shared);
    } else {
      using Kernel =
          typename mxfp6_gemm::grouped::Kernel128x8x128::
              template RebindOutput<cutlass::half_t>;
      launch_array_w6a8<Kernel>(
          output, activation, logical_scales, weight, weight_scales,
          topk_ids, static_cast<int>(n), static_cast<int>(k),
          include_shared);
    }
  }
#endif
}

void array_gemm_w6a8_silu_mxfp8_out_cuda(
    at::Tensor& output,
    at::Tensor& output_scales,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    bool include_shared) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  TORCH_CHECK(false, "mxfp6_torch must be compiled for sm_120a");
#else
  TORCH_CHECK(activation.is_cuda(), "activation must be a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(activation.device());
  auto const device = activation.device();
  TORCH_CHECK(
      (activation.scalar_type() == at::kFloat8_e4m3fn ||
       activation.scalar_type() == at::kByte) &&
          activation.dim() == 2 && activation.is_contiguous(),
      "activation must be contiguous CUDA float8_e4m3fn/uint8 [M,K]");
  check_cuda_byte_tensor(logical_scales, "logical_scales", device);
  check_cuda_byte_tensor(weight, "weight", device);
  check_cuda_byte_tensor(weight_scales, "weight_scales", device);
  check_cuda_byte_tensor(output, "output", device);
  check_cuda_byte_tensor(output_scales, "output_scales", device);
  TORCH_CHECK(
      topk_ids.is_cuda() && topk_ids.device() == device &&
          (topk_ids.scalar_type() == at::kInt ||
           topk_ids.scalar_type() == at::kLong) &&
          topk_ids.is_contiguous() && topk_ids.dim() == 2,
      "topk_ids must be contiguous CUDA int32/int64 [tokens,topk]");
  TORCH_CHECK(weight.dim() == 3, "weight must have shape [E,2N,packed_K]");

  int64_t const tokens = topk_ids.size(0);
  int64_t const topk = topk_ids.size(1);
  int64_t const routes =
      tokens * (topk + static_cast<int64_t>(include_shared));
  int64_t const gate_up = weight.size(1);
  int64_t const intermediate = gate_up / 2;
  int64_t const weight_row_bytes = weight.size(2);
  int64_t const k = activation.size(1);
  TORCH_CHECK(
      tokens > 0 && topk > 0 &&
          routes <= std::numeric_limits<int>::max(),
      "routed rows must be positive and fit int32");
  TORCH_CHECK(
      !include_shared || weight.size(0) >= 2,
      "include_shared requires one trailing shared expert");
  TORCH_CHECK(
      gate_up > 0 && gate_up % 64 == 0 &&
          k > 0 && k % direct_moe::kTileK == 0 &&
          gate_up <= std::numeric_limits<int>::max() &&
          k <= std::numeric_limits<int>::max(),
      "gate/up width must be a multiple of 64 and K a multiple of 128");
  TORCH_CHECK(
      (weight_row_bytes == k ||
       weight_row_bytes * 4 == k * 3),
      "activation K does not match packed weight K");
  TORCH_CHECK(
      activation.size(0) == tokens ||
          activation.size(0) == routes,
      "activation rows must equal tokens or all routed/shared rows");
  TORCH_CHECK(
      logical_scales.dim() == 2 &&
          logical_scales.size(0) == activation.size(0) &&
          logical_scales.size(1) == k / kMxScaleVectorSize,
      "logical_scales must have shape [activation_rows,K/32]");
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) == routes &&
          output.size(1) == intermediate,
      "output must have shape [routes,N]");
  TORCH_CHECK(
      output_scales.dim() == 2 &&
          output_scales.size(0) == routes &&
          output_scales.size(1) ==
              intermediate / kMxScaleVectorSize,
      "output_scales must have shape [routes,N/32]");
  TORCH_CHECK(
      weight_scales.numel() >=
          weight.size(0) * gate_up *
          ((k / 32 + kScaleGroupsPerAtom - 1) /
           kScaleGroupsPerAtom * kScaleGroupsPerAtom),
      "weight_scales is too small");

  int const device_index = activation.get_device();
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(device_index);
  TORCH_CHECK(
      properties.major == 12 && properties.minor == 0,
      "fused array MXFP6 requires SM120");
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  if (topk_ids.scalar_type() == at::kInt) {
    direct_moe::launch_silu_mxfp8<int32_t>(
        output,
        output_scales,
        activation,
        logical_scales,
        weight,
        weight_scales,
        topk_ids,
        include_shared,
        stream.stream());
  } else {
    direct_moe::launch_silu_mxfp8<int64_t>(
        output,
        output_scales,
        activation,
        logical_scales,
        weight,
        weight_scales,
        topk_ids,
        include_shared,
        stream.stream());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

void array_gemm_w6a8_reduce_out_cuda(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    at::Tensor const& topk_weights,
    at::Tensor const& shared_gate,
    std::optional<at::Tensor> shared_output_arg,
    std::optional<at::Tensor> shared_weight_arg,
    std::optional<at::Tensor> shared_weight_scales_arg,
    bool use_packed_vector_loads) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  TORCH_CHECK(false, "mxfp6_torch must be compiled for sm_120a");
#else
  TORCH_CHECK(activation.is_cuda(), "activation must be a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(activation.device());
  auto const device = activation.device();
  TORCH_CHECK(
      (activation.scalar_type() == at::kFloat8_e4m3fn ||
       activation.scalar_type() == at::kByte) &&
          activation.dim() == 2 && activation.is_contiguous(),
      "activation must be contiguous CUDA float8_e4m3fn/uint8 [routes,K]");
  check_cuda_byte_tensor(logical_scales, "logical_scales", device);
  check_cuda_byte_tensor(weight, "weight", device);
  check_cuda_byte_tensor(weight_scales, "weight_scales", device);
  TORCH_CHECK(
      topk_ids.is_cuda() && topk_ids.device() == device &&
          (topk_ids.scalar_type() == at::kInt ||
           topk_ids.scalar_type() == at::kLong) &&
          topk_ids.is_contiguous() && topk_ids.dim() == 2 &&
          topk_ids.size(1) == direct_moe::kReduceMaxRoutes - 1,
      "topk_ids must be contiguous CUDA int32/int64 [tokens,8]");
  TORCH_CHECK(
      topk_weights.is_cuda() &&
          topk_weights.device() == device &&
          topk_weights.scalar_type() == at::kFloat &&
          topk_weights.is_contiguous() &&
          topk_weights.sizes() == topk_ids.sizes(),
      "topk_weights must be contiguous CUDA float32 [tokens,8]");
  TORCH_CHECK(
      shared_gate.is_cuda() &&
          shared_gate.device() == device &&
          (shared_gate.scalar_type() == at::kFloat ||
           shared_gate.scalar_type() == at::kHalf ||
           shared_gate.scalar_type() == at::kBFloat16) &&
          shared_gate.is_contiguous(),
      "shared_gate must be contiguous CUDA float32/float16/bfloat16");
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          (output.scalar_type() == at::kHalf ||
           output.scalar_type() == at::kBFloat16) &&
          output.is_contiguous() && output.dim() == 2,
      "output must be contiguous CUDA float16/bfloat16 [tokens,N]");
  if (shared_output_arg.has_value()) {
    at::Tensor const& shared_output = *shared_output_arg;
    TORCH_CHECK(
        shared_output.is_cuda() &&
            shared_output.device() == device &&
            shared_output.scalar_type() == output.scalar_type() &&
            shared_output.is_contiguous() &&
            shared_output.sizes() == output.sizes(),
        "shared_output must be contiguous and match output's shape, "
        "dtype, and CUDA device");
  }
  TORCH_CHECK(
      weight.dim() == 3,
      "weight must have shape [E,N,packed_K]");
  TORCH_CHECK(
      shared_weight_arg.has_value() ==
          shared_weight_scales_arg.has_value(),
      "shared_weight and shared_weight_scales must be provided together");

  int64_t const tokens = topk_ids.size(0);
  bool const external_shared = shared_output_arg.has_value();
  bool const separate_shared = shared_weight_arg.has_value();
  TORCH_CHECK(
      !(external_shared && separate_shared),
      "separate shared weights cannot be combined with shared_output");
  int64_t const routes =
      tokens * (
          direct_moe::kReduceMaxRoutes -
          static_cast<int64_t>(external_shared));
  int64_t const n = weight.size(1);
  int64_t const weight_row_bytes = weight.size(2);
  int64_t const k = activation.size(1);
  TORCH_CHECK(
      tokens > 0 &&
          tokens <= std::numeric_limits<int>::max(),
      "tokens must be positive and fit int32");
  TORCH_CHECK(
      n > 0 && n % direct_moe::kReduceTileM == 0 &&
          k > 0 && k % direct_moe::kTileK == 0 &&
          n <= std::numeric_limits<int>::max() &&
          k <= std::numeric_limits<int>::max(),
      "N must be a multiple of 16 and K a multiple of 128");
  TORCH_CHECK(
      (weight_row_bytes == k ||
       weight_row_bytes * 4 == k * 3) &&
          activation.size(0) == routes &&
          activation.size(1) == k,
      "activation must have shape [tokens*(8 or 9),K]");
  TORCH_CHECK(
      logical_scales.dim() == 2 &&
          logical_scales.size(0) == routes &&
          logical_scales.size(1) == k / kMxScaleVectorSize,
      "logical_scales must have shape [tokens*(8 or 9),K/32]");
  TORCH_CHECK(
      output.size(0) == tokens && output.size(1) == n,
      "output must have shape [tokens,N]");
  TORCH_CHECK(
      shared_gate.numel() == tokens ||
          (shared_gate.dim() == 2 &&
           shared_gate.size(0) == tokens),
      "shared_gate must contain one row per token");
  TORCH_CHECK(
      weight_scales.numel() >=
          weight.size(0) * n *
          ((k / 32 + kScaleGroupsPerAtom - 1) /
           kScaleGroupsPerAtom * kScaleGroupsPerAtom),
      "weight_scales is too small");
  if (separate_shared) {
    at::Tensor const& shared_weight = *shared_weight_arg;
    at::Tensor const& shared_weight_scales =
        *shared_weight_scales_arg;
    TORCH_CHECK(
        weight.size(0) == 256,
        "separate shared W2 requires 256 routed experts");
    TORCH_CHECK(
        shared_weight.is_cuda() &&
            shared_weight.device() == device &&
            shared_weight.scalar_type() == at::kByte &&
            shared_weight.is_contiguous() &&
            shared_weight.dim() == 3 &&
            shared_weight.size(0) == 1 &&
            shared_weight.size(1) == n &&
            shared_weight.size(2) == weight_row_bytes,
        "shared_weight must be contiguous CUDA uint8 [1,N,packed_K]");
    TORCH_CHECK(
        shared_weight_scales.is_cuda() &&
            shared_weight_scales.device() == device &&
            shared_weight_scales.scalar_type() == at::kByte &&
            shared_weight_scales.is_contiguous() &&
            shared_weight_scales.numel() >=
                n *
                ((k / 32 + kScaleGroupsPerAtom - 1) /
                 kScaleGroupsPerAtom * kScaleGroupsPerAtom),
        "shared_weight_scales is too small for one shared expert");
    TORCH_CHECK(
        output.scalar_type() == at::kBFloat16 &&
            topk_ids.scalar_type() == at::kInt,
        "separate shared W2 currently requires bfloat16 output and int32 ids");
  }

  int const gate_type = shared_gate.scalar_type() == at::kFloat
      ? 1
      : (shared_gate.scalar_type() == at::kHalf ? 2 : 3);
  int const gate_stride =
      shared_gate.numel() == tokens
      ? 1
      : static_cast<int>(shared_gate.size(1));
  int const gate_column =
      shared_gate.numel() == tokens ? 0 : gate_stride - 1;
  int const device_index = activation.get_device();
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(device_index);
  TORCH_CHECK(
      properties.major == 12 && properties.minor == 0,
      "fused array MXFP6 reduce requires SM120");
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  bool const use_qwen_b2_pdl =
      !external_shared &&
      tokens == 2 && n == 2048 && k == 256 &&
      weight.size(0) == (separate_shared ? 256 : 257) &&
      weight_row_bytes == 192 &&
      output.scalar_type() == at::kBFloat16 &&
      topk_ids.scalar_type() == at::kInt;
  bool const use_qwen_b4_packed_vector_loads =
      use_packed_vector_loads && !external_shared &&
      tokens == 4 && n == 2048 && k == 256 &&
      weight.size(0) == (separate_shared ? 256 : 257) &&
      weight_row_bytes == 192 &&
      output.scalar_type() == at::kBFloat16 &&
      topk_ids.scalar_type() == at::kInt &&
      (reinterpret_cast<uintptr_t>(weight.data_ptr()) & 15) == 0 &&
      (!separate_shared ||
       (reinterpret_cast<uintptr_t>(
            shared_weight_arg->data_ptr()) & 15) == 0);
  if (separate_shared) {
    if (use_qwen_b4_packed_vector_loads) {
      direct_moe::launch_reduce<
          int32_t, cutlass::bfloat16_t, false, false, true, true>(
              output,
              activation,
              logical_scales,
              weight,
              weight_scales,
              topk_ids,
              topk_weights,
              shared_gate,
              gate_type,
              gate_stride,
              gate_column,
              stream.stream(),
              nullptr,
              &*shared_weight_arg,
              &*shared_weight_scales_arg);
    } else if (use_qwen_b2_pdl) {
      direct_moe::launch_reduce<
          int32_t, cutlass::bfloat16_t, true, false, true>(
              output,
              activation,
              logical_scales,
              weight,
              weight_scales,
              topk_ids,
              topk_weights,
              shared_gate,
              gate_type,
              gate_stride,
              gate_column,
              stream.stream(),
              nullptr,
              &*shared_weight_arg,
              &*shared_weight_scales_arg);
    } else {
      direct_moe::launch_reduce<
          int32_t, cutlass::bfloat16_t, false, false, true>(
              output,
              activation,
              logical_scales,
              weight,
              weight_scales,
              topk_ids,
              topk_weights,
              shared_gate,
              gate_type,
              gate_stride,
              gate_column,
              stream.stream(),
              nullptr,
              &*shared_weight_arg,
              &*shared_weight_scales_arg);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (output.scalar_type() == at::kBFloat16) {
    if (topk_ids.scalar_type() == at::kInt) {
      if (external_shared) {
        direct_moe::launch_reduce<
            int32_t, cutlass::bfloat16_t, false, true>(
                output,
                activation,
                logical_scales,
                weight,
                weight_scales,
                topk_ids,
                topk_weights,
                shared_gate,
                gate_type,
                gate_stride,
                gate_column,
                stream.stream(),
                &*shared_output_arg);
      } else if (use_qwen_b4_packed_vector_loads) {
        direct_moe::launch_reduce<
            int32_t, cutlass::bfloat16_t,
            false, false, false, true>(
                output,
                activation,
                logical_scales,
                weight,
                weight_scales,
                topk_ids,
                topk_weights,
                shared_gate,
                gate_type,
                gate_stride,
                gate_column,
                stream.stream());
      } else if (use_qwen_b2_pdl) {
        direct_moe::launch_reduce<
            int32_t, cutlass::bfloat16_t, true>(
                output,
                activation,
                logical_scales,
                weight,
                weight_scales,
                topk_ids,
                topk_weights,
                shared_gate,
                gate_type,
                gate_stride,
                gate_column,
                stream.stream());
      } else {
        direct_moe::launch_reduce<
            int32_t, cutlass::bfloat16_t>(
                output,
                activation,
                logical_scales,
                weight,
                weight_scales,
                topk_ids,
                topk_weights,
                shared_gate,
                gate_type,
                gate_stride,
                gate_column,
                stream.stream());
      }
    } else if (external_shared) {
      direct_moe::launch_reduce<
          int64_t, cutlass::bfloat16_t, false, true>(
              output,
              activation,
              logical_scales,
              weight,
              weight_scales,
              topk_ids,
              topk_weights,
              shared_gate,
              gate_type,
              gate_stride,
              gate_column,
              stream.stream(),
              &*shared_output_arg);
    } else {
      direct_moe::launch_reduce<int64_t, cutlass::bfloat16_t>(
          output,
          activation,
          logical_scales,
          weight,
          weight_scales,
          topk_ids,
          topk_weights,
          shared_gate,
          gate_type,
          gate_stride,
          gate_column,
          stream.stream());
    }
  } else if (
      topk_ids.scalar_type() == at::kInt &&
      external_shared) {
    direct_moe::launch_reduce<
        int32_t, cutlass::half_t, false, true>(
            output,
            activation,
            logical_scales,
            weight,
            weight_scales,
            topk_ids,
            topk_weights,
            shared_gate,
            gate_type,
            gate_stride,
            gate_column,
            stream.stream(),
            &*shared_output_arg);
  } else if (topk_ids.scalar_type() == at::kInt) {
    direct_moe::launch_reduce<int32_t, cutlass::half_t>(
        output,
        activation,
        logical_scales,
        weight,
        weight_scales,
        topk_ids,
        topk_weights,
        shared_gate,
        gate_type,
        gate_stride,
        gate_column,
        stream.stream());
  } else if (external_shared) {
    direct_moe::launch_reduce<
        int64_t, cutlass::half_t, false, true>(
            output,
            activation,
            logical_scales,
            weight,
            weight_scales,
            topk_ids,
            topk_weights,
            shared_gate,
            gate_type,
            gate_stride,
            gate_column,
            stream.stream(),
            &*shared_output_arg);
  } else {
    direct_moe::launch_reduce<int64_t, cutlass::half_t>(
        output,
        activation,
        logical_scales,
        weight,
        weight_scales,
        topk_ids,
        topk_weights,
        shared_gate,
        gate_type,
        gate_stride,
        gate_column,
        stream.stream());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

template <class ElementD>
void dispatch_grouped_w6a8(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& activation_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    int n,
    int k,
    int tile_n,
    bool use_pdl) {
  if (tile_n == -11) {
    auto stream = c10::cuda::getCurrentCUDAStream(
        activation.get_device());
    direct_moe::launch_grouped_static<ElementD>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, use_pdl, stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (k == 64) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x8x64::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == -1) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x8x256::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == -9) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel64x8x256Pingpong::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == -10) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel64x8x128CooperativeStage4::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == -3) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x8x256Stage3::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == -8) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel256x8x128::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == -13) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel256x8x64::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == -14) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x8x64::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == -4) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x8x128Stage4::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == -6) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x8x128Stage6::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == 16) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x16x128Pingpong::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == 32) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x32x128::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == 64) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x64x128::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  if (tile_n == 128) {
    using Kernel =
        typename mxfp6_gemm::grouped::Kernel128x128x128::
            template RebindOutput<ElementD>;
    launch_grouped_w6a8<Kernel>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, n, k, use_pdl);
    return;
  }
  using Kernel =
      typename mxfp6_gemm::grouped::Kernel128x8x128::
          template RebindOutput<ElementD>;
  launch_grouped_w6a8<Kernel>(
      output, activation, activation_scales, weight, weight_scales,
      expert_offsets, scale_offsets, n, k, use_pdl);
}

void grouped_gemm_w6a8_out_cuda(
    at::Tensor& output,
    at::Tensor const& activation,
    at::Tensor const& activation_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& expert_offsets,
    at::Tensor const& scale_offsets,
    int64_t tile_n,
    bool use_pdl) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  TORCH_CHECK(false, "mxfp6_torch must be compiled for sm_120a");
#else
  TORCH_CHECK(activation.is_cuda(), "activation must be a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(activation.device());
  auto const device = activation.device();
  TORCH_CHECK(
      activation.scalar_type() == at::kFloat8_e4m3fn ||
          activation.scalar_type() == at::kByte,
      "activation must be float8_e4m3fn or uint8");
  TORCH_CHECK(
      activation.dim() == 2 && activation.is_contiguous(),
      "activation must be contiguous [M,K]");
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          output.is_contiguous() && output.dim() == 2,
      "output must be a contiguous CUDA [M,N] tensor on the input device");
  TORCH_CHECK(
      output.scalar_type() == at::kHalf ||
          output.scalar_type() == at::kBFloat16,
      "output must be float16 or bfloat16");
  check_cuda_byte_tensor(activation_scales, "activation_scales", device);
  check_cuda_byte_tensor(weight, "weight", device);
  check_cuda_byte_tensor(weight_scales, "weight_scales", device);
  TORCH_CHECK(
      expert_offsets.is_cuda() && expert_offsets.device() == device &&
          expert_offsets.scalar_type() == at::kLong &&
          expert_offsets.is_contiguous() && expert_offsets.dim() == 1,
      "expert_offsets must be a contiguous CUDA int64 tensor");
  TORCH_CHECK(
      scale_offsets.is_cuda() && scale_offsets.device() == device &&
          scale_offsets.scalar_type() == at::kLong &&
          scale_offsets.is_contiguous() && scale_offsets.dim() == 1,
      "scale_offsets must be a contiguous CUDA int64 tensor");
  TORCH_CHECK(weight.dim() == 3, "weight must have shape [E,N,packed_K]");

  int64_t const num_experts = weight.size(0);
  int64_t const n = weight.size(1);
  int64_t const packed_k = weight.size(2);
  int64_t const k = packed_k * 4 / 3;
  int64_t const m = activation.size(0);
  TORCH_CHECK(
      num_experts > 0 &&
          num_experts <= std::numeric_limits<int>::max(),
      "number of experts must fit int32");
  TORCH_CHECK(
      m > 0 && m <= std::numeric_limits<int>::max(),
      "activation rows must be positive and fit int32");
  TORCH_CHECK(n > 0 && n % 128 == 0, "N must be a multiple of 128");
  TORCH_CHECK(
      k > 0 && k % kMxScaleVectorSize == 0,
      "K must be a multiple of 32");
  TORCH_CHECK(
      packed_k * 4 == k * 3,
      "packed weight K dimension is not a valid FP6 representation");
  TORCH_CHECK(
      activation.size(1) == k,
      "activation K does not match packed weight K");
  TORCH_CHECK(
      output.size(0) == m && output.size(1) == n,
      "output shape must be [M,N]");
  TORCH_CHECK(
      expert_offsets.numel() == num_experts + 1 &&
          scale_offsets.numel() == num_experts + 1,
      "offset tensors must contain E+1 entries");
  TORCH_CHECK(
      weight_scales.numel() >=
          num_experts * n *
          ((k / 32 + kScaleGroupsPerAtom - 1) /
           kScaleGroupsPerAtom * kScaleGroupsPerAtom),
      "weight_scales is too small for the packed scale atom");
  TORCH_CHECK(
      tile_n == -14 || tile_n == -13 ||
          tile_n == -11 || tile_n == -10 || tile_n == -9 ||
          tile_n == -8 || tile_n == -6 || tile_n == -4 ||
          tile_n == -3 || tile_n == -1 || tile_n == 0 || tile_n == 8 ||
          tile_n == 16 || tile_n == 32 ||
          tile_n == 64 || tile_n == 128,
      "tile_n must be -14, -13, -11, -10, "
      "-9, -8, -6, -4, -3, -1, 0, 8, 16, 32, 64, or 128");

  int const device_index = activation.get_device();
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(device_index);
  TORCH_CHECK(
      properties.major == 12 && properties.minor == 0,
      "grouped MXFP6 requires SM120; current device is SM",
      properties.major, properties.minor);

  int selected_tile_n = static_cast<int>(tile_n);
  if (selected_tile_n == 0) {
    int64_t const active_expert_bound = std::min(m, num_experts);
    int64_t const average_rows =
        (m + active_expert_bound - 1) / active_expert_bound;
    selected_tile_n =
        average_rows >= 96
        ? 128
        : (average_rows >= 48
               ? 64
               : (average_rows >= 24
                      ? 32
                      : (average_rows >= 12 ? 16 : 8)));
  }
  if (output.scalar_type() == at::kBFloat16) {
    dispatch_grouped_w6a8<cutlass::bfloat16_t>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, static_cast<int>(n),
        static_cast<int>(k), selected_tile_n, use_pdl);
  } else {
    dispatch_grouped_w6a8<cutlass::half_t>(
        output, activation, activation_scales, weight, weight_scales,
        expert_offsets, scale_offsets, static_cast<int>(n),
        static_cast<int>(k), selected_tile_n, use_pdl);
  }
#endif
}

template <class Element>
__device__ __forceinline__ float load_element(Element const* pointer) {
  return static_cast<float>(*pointer);
}

__device__ __forceinline__ float load_gate(
    void const* gate,
    int gate_type,
    int64_t index) {
  if (gate_type == 1) {
    return static_cast<float const*>(gate)[index];
  }
  if (gate_type == 2) {
    return static_cast<float>(
        static_cast<at::Half const*>(gate)[index]);
  }
  return static_cast<float>(
      static_cast<at::BFloat16 const*>(gate)[index]);
}

namespace tp2_fused_reduce {

constexpr int kMaxBlocks = 36;
constexpr int kWorldSize = 2;
constexpr int kThreads = 512;
constexpr int kPackValues = 8;
using FlagType = uint32_t;

struct Signal {
  alignas(128) FlagType start[kMaxBlocks][8];
  alignas(128) FlagType end[kMaxBlocks][8];
  alignas(128) FlagType flag[kMaxBlocks];
};

struct alignas(16) RankSignals {
  Signal* signals[8];
};

struct alignas(16) RankBuffers {
  __nv_bfloat16* pointers[8];
};

struct alignas(16) Pack {
  __nv_bfloat16 values[kPackValues];
};

__device__ __forceinline__ void store_release(
    FlagType* address,
    FlagType value) {
  asm volatile(
      "st.release.sys.global.u32 [%1], %0;"
      :
      : "r"(value), "l"(address));
}

__device__ __forceinline__ FlagType load_acquire(
    FlagType* address) {
  FlagType value;
  asm volatile(
      "ld.acquire.sys.global.u32 %0, [%1];"
      : "=r"(value)
      : "l"(address));
  return value;
}

__device__ __forceinline__ void store_volatile(
    FlagType* address,
    FlagType value) {
  asm volatile(
      "st.volatile.global.u32 [%1], %0;"
      :
      : "r"(value), "l"(address));
}

__device__ __forceinline__ FlagType load_volatile(
    FlagType* address) {
  FlagType value;
  asm volatile(
      "ld.volatile.global.u32 %0, [%1];"
      : "=r"(value)
      : "l"(address));
  return value;
}

__device__ __forceinline__ void data_barrier(
    RankSignals const& signals,
    Signal* self,
    int rank) {
  __syncthreads();
  FlagType const value = self->flag[blockIdx.x] + 1;
  if (threadIdx.x < kWorldSize) {
    auto* peer =
        &signals.signals[threadIdx.x]
             ->start[blockIdx.x][rank];
    auto* local = &self->start[blockIdx.x][threadIdx.x];
    store_release(peer, value);
    while (load_acquire(local) != value) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    self->flag[blockIdx.x] = value;
  }
}

__device__ __forceinline__ void final_barrier(
    RankSignals const& signals,
    Signal* self,
    int rank) {
  __syncthreads();
  FlagType const value = self->flag[blockIdx.x] + 1;
  if (threadIdx.x < kWorldSize) {
    auto* peer =
        &signals.signals[threadIdx.x]
             ->end[blockIdx.x][rank];
    auto* local = &self->end[blockIdx.x][threadIdx.x];
    store_volatile(peer, value);
    while (load_volatile(local) != value) {
    }
  }
  if (threadIdx.x == 0) {
    self->flag[blockIdx.x] = value;
  }
}

__global__ __launch_bounds__(kThreads, 1)
void kernel(
    at::BFloat16* output,
    at::BFloat16 const* routed_output,
    float const* topk_weights,
    int32_t const* inverse_permutation,
    at::BFloat16 const* shared_output,
    void const* shared_gate,
    int gate_type,
    int gate_stride,
    int gate_column,
    int tokens,
    int topk,
    int hidden_size,
    int routed_rows,
    RankSignals signals,
    RankBuffers buffers,
    int rank) {
  __shared__ int32_t routed_indices[2][8];
  __shared__ float routed_weights[2][8];
  __shared__ float shared_scales[2];

  int const packs_per_token = hidden_size / kPackValues;
  int const total_packs = tokens * packs_per_token;
  int const block_stride = gridDim.x * blockDim.x;
  auto* local_packs = reinterpret_cast<Pack*>(
      buffers.pointers[rank]);

  for (int base = blockIdx.x * blockDim.x;
       base < total_packs;
       base += block_stride) {
    int const first_token = base / packs_per_token;
    int const chunk_tokens = min(2, tokens - first_token);
    if (threadIdx.x < chunk_tokens * topk) {
      int const local_token = threadIdx.x / topk;
      int const route = threadIdx.x - local_token * topk;
      int64_t const route_index =
          static_cast<int64_t>(first_token + local_token) * topk +
          route;
      routed_indices[local_token][route] =
          inverse_permutation[route_index];
      routed_weights[local_token][route] =
          topk_weights[route_index];
    }
    if (threadIdx.x < chunk_tokens) {
      int const token = first_token + threadIdx.x;
      float const gate = load_gate(
          shared_gate,
          gate_type,
          static_cast<int64_t>(token) * gate_stride + gate_column);
      shared_scales[threadIdx.x] = 1.0f / (1.0f + expf(-gate));
    }
    __syncthreads();

    int const pack_index = base + threadIdx.x;
    if (pack_index < total_packs) {
      int const token = pack_index / packs_per_token;
      int const local_token = token - first_token;
      int const column =
          (pack_index - token * packs_per_token) * kPackValues;
      float accumulators[kPackValues];
      auto const shared_pack = *reinterpret_cast<Pack const*>(
          shared_output +
          static_cast<int64_t>(token) * hidden_size + column);
#pragma unroll
      for (int value = 0; value < kPackValues; ++value) {
        accumulators[value] = shared_scales[local_token] *
            __bfloat162float(shared_pack.values[value]);
      }
#pragma unroll
      for (int route = 0; route < 8; ++route) {
        int32_t const routed_row =
            routed_indices[local_token][route];
        if (routed_row >= 0 && routed_row < routed_rows) {
          auto const routed_pack = *reinterpret_cast<Pack const*>(
              routed_output +
              static_cast<int64_t>(routed_row) * hidden_size +
              column);
          float const weight = routed_weights[local_token][route];
#pragma unroll
          for (int value = 0; value < kPackValues; ++value) {
            accumulators[value] += weight *
                __bfloat162float(routed_pack.values[value]);
          }
        }
      }
      Pack local_pack;
#pragma unroll
      for (int value = 0; value < kPackValues; ++value) {
        local_pack.values[value] =
            __float2bfloat16(accumulators[value]);
      }
      local_packs[pack_index] = local_pack;
    }
    __syncthreads();
  }

  data_barrier(signals, signals.signals[rank], rank);

  auto const* rank0 = reinterpret_cast<Pack const*>(
      buffers.pointers[0]);
  auto const* rank1 = reinterpret_cast<Pack const*>(
      buffers.pointers[1]);
  auto* output_packs = reinterpret_cast<Pack*>(output);
  for (int pack_index = blockIdx.x * blockDim.x + threadIdx.x;
       pack_index < total_packs;
       pack_index += gridDim.x * blockDim.x) {
    Pack const first = rank0[pack_index];
    Pack const second = rank1[pack_index];
    Pack result;
#pragma unroll
    for (int value = 0; value < kPackValues; ++value) {
      result.values[value] = __float2bfloat16(
          __bfloat162float(first.values[value]) +
          __bfloat162float(second.values[value]));
    }
    output_packs[pack_index] = result;
  }

  final_barrier(signals, signals.signals[rank], rank);
}

}  // namespace tp2_fused_reduce

template <class Element, bool UsePdl = false>
__global__ void moe_reduce_kernel(
    Element* output,
    Element const* routed_output,
    float const* topk_weights,
    int32_t const* inverse_permutation,
    Element const* shared_output,
    void const* shared_gate,
    int gate_type,
    int gate_stride,
    int gate_column,
    int topk,
    int hidden_size,
    int routed_rows) {
  if constexpr (UsePdl) {
    cutlass::arch::wait_on_dependent_grids();
  }
  constexpr int kMaxTopK = 64;
  int const token = blockIdx.x;
  int const column = blockIdx.y * blockDim.x + threadIdx.x;
  __shared__ int32_t routed_indices[kMaxTopK];
  __shared__ float routed_weights[kMaxTopK];
  __shared__ float shared_scale;

  for (int index = threadIdx.x; index < topk; index += blockDim.x) {
    int64_t const route = static_cast<int64_t>(token) * topk + index;
    routed_indices[index] = inverse_permutation[route];
    routed_weights[index] = topk_weights[route];
  }
  if (threadIdx.x == 0) {
    if (shared_output == nullptr) {
      shared_scale = 0.0f;
    } else if (shared_gate == nullptr) {
      shared_scale = 1.0f;
    } else {
      float const gate = load_gate(
          shared_gate,
          gate_type,
          static_cast<int64_t>(token) * gate_stride + gate_column);
      shared_scale = 1.0f / (1.0f + expf(-gate));
    }
  }
  __syncthreads();

  if (column < hidden_size) {
    float accumulator = 0.0f;
    if (shared_output != nullptr) {
      accumulator = shared_scale * load_element(
          shared_output +
          static_cast<int64_t>(token) * hidden_size + column);
    }
#pragma unroll 1
    for (int index = 0; index < topk; ++index) {
      int32_t const routed_row = routed_indices[index];
      if (routed_row >= 0 && routed_row < routed_rows) {
        accumulator += routed_weights[index] * load_element(
            routed_output +
            static_cast<int64_t>(routed_row) * hidden_size + column);
      }
    }
    output[static_cast<int64_t>(token) * hidden_size + column] =
        static_cast<Element>(accumulator);
  }
}

template <int Values>
struct alignas(Values * sizeof(__nv_bfloat16)) Bfloat16Vector {
  __nv_bfloat16 values[Values];
};

template <bool UsePdl, int Values>
__global__ __launch_bounds__(128, 4)
void moe_reduce_bfloat16x2_kernel(
    at::BFloat16* output,
    at::BFloat16 const* routed_output,
    float const* topk_weights,
    int32_t const* inverse_permutation,
    at::BFloat16 const* shared_output,
    void const* shared_gate,
    int gate_type,
    int gate_stride,
    int gate_column,
    int hidden_size,
    int routed_rows) {
  if constexpr (UsePdl) {
    cutlass::arch::wait_on_dependent_grids();
  }
  constexpr int kTopK = 8;
  int const token = static_cast<int>(blockIdx.x);
  int const vector_index =
      static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
  int const column = vector_index * Values;
  __shared__ int32_t routed_indices[kTopK];
  __shared__ float routed_weights[kTopK];
  __shared__ float shared_scale;

  if (threadIdx.x < kTopK) {
    int64_t const route =
        static_cast<int64_t>(token) * kTopK + threadIdx.x;
    routed_indices[threadIdx.x] = inverse_permutation[route];
    routed_weights[threadIdx.x] = topk_weights[route];
  }
  if (threadIdx.x == 0) {
    if (shared_output == nullptr) {
      shared_scale = 0.0f;
    } else if (shared_gate == nullptr) {
      shared_scale = 1.0f;
    } else {
      float const gate = load_gate(
          shared_gate,
          gate_type,
          static_cast<int64_t>(token) * gate_stride + gate_column);
      shared_scale = 1.0f / (1.0f + expf(-gate));
    }
  }
  __syncthreads();

  if (column < hidden_size) {
    float accumulators[Values]{};
    if (shared_output != nullptr) {
      auto const shared_vector =
          *reinterpret_cast<Bfloat16Vector<Values> const*>(
              shared_output +
              static_cast<int64_t>(token) * hidden_size + column);
      CUTE_UNROLL
      for (int value = 0; value < Values; ++value) {
        accumulators[value] = shared_scale *
            __bfloat162float(shared_vector.values[value]);
      }
    }
    CUTE_UNROLL
    for (int index = 0; index < kTopK; ++index) {
      int32_t const routed_row = routed_indices[index];
      if (routed_row >= 0 && routed_row < routed_rows) {
        auto const routed_vector =
            *reinterpret_cast<Bfloat16Vector<Values> const*>(
                routed_output +
                static_cast<int64_t>(routed_row) * hidden_size + column);
        float const route_weight = routed_weights[index];
        CUTE_UNROLL
        for (int value = 0; value < Values; ++value) {
          accumulators[value] += route_weight *
              __bfloat162float(routed_vector.values[value]);
        }
      }
    }
    Bfloat16Vector<Values> result;
    CUTE_UNROLL
    for (int value = 0; value < Values; ++value) {
      result.values[value] = __float2bfloat16(accumulators[value]);
    }
    *reinterpret_cast<Bfloat16Vector<Values>*>(
        output + static_cast<int64_t>(token) * hidden_size + column) =
            result;
  }
}

template <int Values>
void launch_moe_reduce_bfloat16_vector(
    at::BFloat16* output,
    at::BFloat16 const* routed_output,
    float const* topk_weights,
    int32_t const* inverse_permutation,
    at::BFloat16 const* shared_output,
    void const* shared_gate,
    int gate_type,
    int gate_stride,
    int gate_column,
    int tokens,
    int hidden_size,
    int routed_rows,
    cudaStream_t stream,
    bool use_pdl) {
  constexpr int kThreads = 128;
  dim3 const grid(
      static_cast<unsigned>(tokens),
      static_cast<unsigned>(
          (hidden_size / Values + kThreads - 1) / kThreads));
  if (use_pdl) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(kThreads);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id =
        cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        moe_reduce_bfloat16x2_kernel<true, Values>,
        output,
        routed_output,
        topk_weights,
        inverse_permutation,
        shared_output,
        shared_gate,
        gate_type,
        gate_stride,
        gate_column,
        hidden_size,
        routed_rows));
  } else {
    moe_reduce_bfloat16x2_kernel<false, Values>
        <<<grid, kThreads, 0, stream>>>(
            output,
            routed_output,
            topk_weights,
            inverse_permutation,
            shared_output,
            shared_gate,
            gate_type,
            gate_stride,
            gate_column,
            hidden_size,
            routed_rows);
  }
}

void launch_moe_reduce_bfloat16x2(
    at::BFloat16* output,
    at::BFloat16 const* routed_output,
    float const* topk_weights,
    int32_t const* inverse_permutation,
    at::BFloat16 const* shared_output,
    void const* shared_gate,
    int gate_type,
    int gate_stride,
    int gate_column,
    int tokens,
    int hidden_size,
    int routed_rows,
    cudaStream_t stream,
    bool use_pdl) {
  if (tokens >= 64 && grouped_reduce_x4()) {
    launch_moe_reduce_bfloat16_vector<4>(
        output, routed_output, topk_weights, inverse_permutation,
        shared_output, shared_gate, gate_type, gate_stride, gate_column,
        tokens, hidden_size, routed_rows, stream, use_pdl);
  } else {
    launch_moe_reduce_bfloat16_vector<2>(
        output, routed_output, topk_weights, inverse_permutation,
        shared_output, shared_gate, gate_type, gate_stride, gate_column,
        tokens, hidden_size, routed_rows, stream, use_pdl);
  }
}

template <class Element>
void launch_moe_reduce(
    Element* output,
    Element const* routed_output,
    float const* topk_weights,
    int32_t const* inverse_permutation,
    Element const* shared_output,
    void const* shared_gate,
    int gate_type,
    int gate_stride,
    int gate_column,
    int topk,
    int hidden_size,
    int routed_rows,
    dim3 grid,
    int threads,
    cudaStream_t stream,
    bool use_pdl) {
  if (use_pdl) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(threads);
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id =
        cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        moe_reduce_kernel<Element, true>,
        output,
        routed_output,
        topk_weights,
        inverse_permutation,
        shared_output,
        shared_gate,
        gate_type,
        gate_stride,
        gate_column,
        topk,
        hidden_size,
        routed_rows));
  } else {
    moe_reduce_kernel<Element, false>
        <<<grid, threads, 0, stream>>>(
            output,
            routed_output,
            topk_weights,
            inverse_permutation,
            shared_output,
            shared_gate,
            gate_type,
            gate_stride,
            gate_column,
            topk,
            hidden_size,
            routed_rows);
  }
}

template <class Element>
__global__ void moe_reduce_array_kernel(
    Element* output,
    Element const* combined_output,
    float const* topk_weights,
    void const* shared_gate,
    int gate_type,
    int gate_stride,
    int gate_column,
    int topk,
    int hidden_size) {
  constexpr int kMaxTopK = 64;
  int const token = blockIdx.x;
  int const column = blockIdx.y * blockDim.x + threadIdx.x;
  __shared__ float routed_weights[kMaxTopK];
  __shared__ float shared_scale;

  for (int index = threadIdx.x; index < topk; index += blockDim.x) {
    routed_weights[index] =
        topk_weights[static_cast<int64_t>(token) * topk + index];
  }
  if (threadIdx.x == 0) {
    float const gate = load_gate(
        shared_gate,
        gate_type,
        static_cast<int64_t>(token) * gate_stride + gate_column);
    shared_scale = 1.0f / (1.0f + expf(-gate));
  }
  __syncthreads();

  int64_t const first_row =
      static_cast<int64_t>(token) * (topk + 1);
  if (column < hidden_size) {
    float accumulator = shared_scale * load_element(
        combined_output +
        (first_row + topk) * hidden_size + column);
#pragma unroll 1
    for (int index = 0; index < topk; ++index) {
      accumulator += routed_weights[index] * load_element(
          combined_output +
          (first_row + index) * hidden_size + column);
    }
    output[static_cast<int64_t>(token) * hidden_size + column] =
        static_cast<Element>(accumulator);
  }
}

void moe_reduce_array_out_cuda(
    at::Tensor& output,
    at::Tensor const& combined_output,
    at::Tensor const& topk_weights,
    at::Tensor const& shared_gate) {
  TORCH_CHECK(
      combined_output.is_cuda() && combined_output.is_contiguous() &&
          combined_output.dim() == 2 &&
          (combined_output.scalar_type() == at::kHalf ||
           combined_output.scalar_type() == at::kBFloat16),
      "combined_output must be contiguous CUDA float16/bfloat16 "
      "[tokens*(topk+1),H]");
  c10::cuda::CUDAGuard device_guard(combined_output.device());
  auto const device = combined_output.device();
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.device() == device &&
          topk_weights.scalar_type() == at::kFloat &&
          topk_weights.is_contiguous() && topk_weights.dim() == 2,
      "topk_weights must be contiguous CUDA float32 [tokens,topk]");
  int64_t const tokens = topk_weights.size(0);
  int64_t const topk = topk_weights.size(1);
  int64_t const hidden_size = combined_output.size(1);
  TORCH_CHECK(
      shared_gate.is_cuda() && shared_gate.device() == device &&
          (shared_gate.scalar_type() == at::kFloat ||
           shared_gate.scalar_type() == at::kHalf ||
           shared_gate.scalar_type() == at::kBFloat16) &&
          shared_gate.is_contiguous() &&
          (shared_gate.numel() == tokens ||
           (shared_gate.dim() == 2 &&
            shared_gate.size(0) == tokens)),
      "shared_gate must be contiguous CUDA float32/float16/bfloat16");

  TORCH_CHECK(
      tokens > 0 && tokens <= std::numeric_limits<int>::max() &&
          topk > 0 && topk <= 64 &&
          hidden_size > 0 &&
          hidden_size <= std::numeric_limits<int>::max(),
      "array MoE reduce dimensions must fit int32 and topk <= 64");
  TORCH_CHECK(
      combined_output.size(0) == tokens * (topk + 1),
      "combined_output must contain topk routed rows and one shared row "
      "per token");
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          output.scalar_type() == combined_output.scalar_type() &&
          output.is_contiguous() && output.dim() == 2 &&
          output.size(0) == tokens && output.size(1) == hidden_size,
      "output must be contiguous [tokens,H] with combined_output's dtype");

  int const gate_type = shared_gate.scalar_type() == at::kFloat
      ? 1
      : (shared_gate.scalar_type() == at::kHalf ? 2 : 3);
  int const gate_stride =
      shared_gate.numel() == tokens
      ? 1
      : static_cast<int>(shared_gate.size(1));
  int const gate_column =
      shared_gate.numel() == tokens ? 0 : gate_stride - 1;
  int const device_index = combined_output.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  constexpr int threads = 256;
  dim3 const grid(
      static_cast<unsigned>(tokens),
      static_cast<unsigned>((hidden_size + threads - 1) / threads));
  if (combined_output.scalar_type() == at::kHalf) {
    moe_reduce_array_kernel<at::Half>
        <<<grid, threads, 0, stream.stream()>>>(
            output.data_ptr<at::Half>(),
            combined_output.data_ptr<at::Half>(),
            topk_weights.data_ptr<float>(),
            shared_gate.data_ptr(),
            gate_type,
            gate_stride,
            gate_column,
            static_cast<int>(topk),
            static_cast<int>(hidden_size));
  } else {
    moe_reduce_array_kernel<at::BFloat16>
        <<<grid, threads, 0, stream.stream()>>>(
            output.data_ptr<at::BFloat16>(),
            combined_output.data_ptr<at::BFloat16>(),
            topk_weights.data_ptr<float>(),
            shared_gate.data_ptr(),
            gate_type,
            gate_stride,
            gate_column,
            static_cast<int>(topk),
            static_cast<int>(hidden_size));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_reduce_out_cuda(
    at::Tensor& output,
    at::Tensor const& routed_output,
    at::Tensor const& topk_weights,
    at::Tensor const& inverse_permutation,
    std::optional<at::Tensor> shared_output_arg,
    std::optional<at::Tensor> shared_gate_arg,
    bool use_pdl) {
  TORCH_CHECK(
      routed_output.is_cuda() && routed_output.is_contiguous() &&
          routed_output.dim() == 2 &&
          (routed_output.scalar_type() == at::kHalf ||
           routed_output.scalar_type() == at::kBFloat16),
      "routed_output must be contiguous CUDA float16/bfloat16 [routes,H]");
  c10::cuda::CUDAGuard device_guard(routed_output.device());
  auto const device = routed_output.device();
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.device() == device &&
          topk_weights.scalar_type() == at::kFloat &&
          topk_weights.is_contiguous() && topk_weights.dim() == 2,
      "topk_weights must be contiguous CUDA float32 [tokens,topk]");
  TORCH_CHECK(
      inverse_permutation.is_cuda() &&
          inverse_permutation.device() == device &&
          inverse_permutation.scalar_type() == at::kInt &&
          inverse_permutation.is_contiguous() &&
          inverse_permutation.dim() == 1,
      "inverse_permutation must be a contiguous CUDA int32 vector");

  int64_t const tokens = topk_weights.size(0);
  int64_t const topk = topk_weights.size(1);
  int64_t const routed_rows = routed_output.size(0);
  int64_t const hidden_size = routed_output.size(1);
  TORCH_CHECK(
      tokens > 0 && tokens <= std::numeric_limits<int>::max() &&
          topk > 0 && topk <= 64 &&
          hidden_size > 0 && hidden_size <= std::numeric_limits<int>::max() &&
          routed_rows > 0 &&
          routed_rows <= std::numeric_limits<int>::max(),
      "MoE reduce dimensions must be positive, fit int32, and topk <= 64");
  TORCH_CHECK(
      inverse_permutation.numel() == tokens * topk,
      "inverse_permutation must contain tokens*topk entries");
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          output.scalar_type() == routed_output.scalar_type() &&
          output.is_contiguous() && output.dim() == 2 &&
          output.size(0) == tokens && output.size(1) == hidden_size,
      "output must be contiguous [tokens,H] with routed_output's dtype");

  void const* shared_output = nullptr;
  if (shared_output_arg.has_value()) {
    at::Tensor const& shared = *shared_output_arg;
    TORCH_CHECK(
        shared.is_cuda() && shared.device() == device &&
            shared.scalar_type() == routed_output.scalar_type() &&
            shared.is_contiguous() && shared.dim() == 2 &&
            shared.size(0) == tokens && shared.size(1) == hidden_size,
        "shared_output must be contiguous [tokens,H] with routed_output's "
        "dtype");
    shared_output = shared.data_ptr();
  }

  void const* shared_gate = nullptr;
  int gate_type = 0;
  int gate_stride = 1;
  int gate_column = 0;
  if (shared_gate_arg.has_value()) {
    at::Tensor const& gate = *shared_gate_arg;
    TORCH_CHECK(
        shared_output != nullptr,
        "shared_gate requires shared_output");
    TORCH_CHECK(
        gate.is_cuda() && gate.device() == device &&
            (gate.scalar_type() == at::kFloat ||
             gate.scalar_type() == at::kHalf ||
             gate.scalar_type() == at::kBFloat16) &&
            gate.is_contiguous() &&
            (gate.numel() == tokens ||
             (gate.dim() == 2 && gate.size(0) == tokens)),
        "shared_gate must be contiguous CUDA float32/float16/bfloat16 with "
        "one value or one row per token");
    shared_gate = gate.data_ptr();
    gate_type = gate.scalar_type() == at::kFloat
        ? 1
        : (gate.scalar_type() == at::kHalf ? 2 : 3);
    if (gate.numel() != tokens) {
      gate_stride = static_cast<int>(gate.size(1));
      gate_column = gate_stride - 1;
    }
  }

  int const device_index = routed_output.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  constexpr int threads = 256;
  dim3 const grid(
      static_cast<unsigned>(tokens),
      static_cast<unsigned>((hidden_size + threads - 1) / threads));
  if (routed_output.scalar_type() == at::kHalf) {
    launch_moe_reduce(
        output.data_ptr<at::Half>(),
        routed_output.data_ptr<at::Half>(),
        topk_weights.data_ptr<float>(),
        inverse_permutation.data_ptr<int32_t>(),
        static_cast<at::Half const*>(shared_output),
        shared_gate,
        gate_type,
        gate_stride,
        gate_column,
        static_cast<int>(topk),
        static_cast<int>(hidden_size),
        static_cast<int>(routed_rows),
        grid,
        threads,
        stream.stream(),
        use_pdl);
  } else {
    if (topk == 8 && hidden_size % 2 == 0) {
      launch_moe_reduce_bfloat16x2(
          output.data_ptr<at::BFloat16>(),
          routed_output.data_ptr<at::BFloat16>(),
          topk_weights.data_ptr<float>(),
          inverse_permutation.data_ptr<int32_t>(),
          static_cast<at::BFloat16 const*>(shared_output),
          shared_gate,
          gate_type,
          gate_stride,
          gate_column,
          static_cast<int>(tokens),
          static_cast<int>(hidden_size),
          static_cast<int>(routed_rows),
          stream.stream(),
          use_pdl);
    } else {
      launch_moe_reduce(
          output.data_ptr<at::BFloat16>(),
          routed_output.data_ptr<at::BFloat16>(),
          topk_weights.data_ptr<float>(),
          inverse_permutation.data_ptr<int32_t>(),
          static_cast<at::BFloat16 const*>(shared_output),
          shared_gate,
          gate_type,
          gate_stride,
          gate_column,
          static_cast<int>(topk),
          static_cast<int>(hidden_size),
          static_cast<int>(routed_rows),
          grid,
          threads,
          stream.stream(),
          use_pdl);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_reduce_tp2_out_cuda(
    at::Tensor& output,
    at::Tensor const& routed_output,
    at::Tensor const& topk_weights,
    at::Tensor const& inverse_permutation,
    at::Tensor const& shared_output,
    at::Tensor const& shared_gate,
    std::vector<int64_t> const& signal_ptrs,
    std::vector<int64_t> const& buffer_ptrs,
    int64_t rank) {
  TORCH_CHECK(
      routed_output.is_cuda() && routed_output.is_contiguous() &&
          routed_output.scalar_type() == at::kBFloat16 &&
          routed_output.dim() == 2,
      "routed_output must be contiguous CUDA bfloat16 [routes,H]");
  c10::cuda::CUDAGuard device_guard(routed_output.device());
  auto const device = routed_output.device();
  TORCH_CHECK(
      topk_weights.is_cuda() && topk_weights.device() == device &&
          topk_weights.scalar_type() == at::kFloat &&
          topk_weights.is_contiguous() && topk_weights.dim() == 2 &&
          topk_weights.size(1) == 8,
      "topk_weights must be contiguous CUDA float32 [tokens,8]");
  TORCH_CHECK(
      inverse_permutation.is_cuda() &&
          inverse_permutation.device() == device &&
          inverse_permutation.scalar_type() == at::kInt &&
          inverse_permutation.is_contiguous() &&
          inverse_permutation.numel() == topk_weights.numel(),
      "inverse_permutation must be contiguous CUDA int32 [tokens*8]");
  int64_t const tokens = topk_weights.size(0);
  int64_t const hidden_size = routed_output.size(1);
  TORCH_CHECK(
      tokens > 0 && tokens <= std::numeric_limits<int>::max() &&
          hidden_size > 0 && hidden_size % 8 == 0 &&
          hidden_size <= std::numeric_limits<int>::max() &&
          routed_output.size(0) <= std::numeric_limits<int>::max(),
      "TP2 fused reduce dimensions must fit int32 and H must be divisible by 8");
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          output.scalar_type() == at::kBFloat16 &&
          output.is_contiguous() &&
          output.sizes() == at::IntArrayRef({tokens, hidden_size}),
      "output must be contiguous CUDA bfloat16 [tokens,H]");
  TORCH_CHECK(
      shared_output.is_cuda() && shared_output.device() == device &&
          shared_output.scalar_type() == at::kBFloat16 &&
          shared_output.is_contiguous() &&
          shared_output.sizes() == output.sizes(),
      "shared_output must match output");
  TORCH_CHECK(
      shared_gate.is_cuda() && shared_gate.device() == device &&
          (shared_gate.scalar_type() == at::kFloat ||
           shared_gate.scalar_type() == at::kHalf ||
           shared_gate.scalar_type() == at::kBFloat16) &&
          shared_gate.is_contiguous() &&
          (shared_gate.numel() == tokens ||
           (shared_gate.dim() == 2 && shared_gate.size(0) == tokens)),
      "shared_gate must contain one value or row per token");
  TORCH_CHECK(
      signal_ptrs.size() == tp2_fused_reduce::kWorldSize &&
          buffer_ptrs.size() == tp2_fused_reduce::kWorldSize &&
          (rank == 0 || rank == 1),
      "TP2 fused reduce requires two signal/buffer pointers and rank 0 or 1");

  tp2_fused_reduce::RankSignals signals{};
  tp2_fused_reduce::RankBuffers buffers{};
  for (int index = 0;
       index < tp2_fused_reduce::kWorldSize;
       ++index) {
    TORCH_CHECK(
        signal_ptrs[index] != 0 && buffer_ptrs[index] != 0,
        "TP2 fused reduce pointers must be nonzero");
    signals.signals[index] =
        reinterpret_cast<tp2_fused_reduce::Signal*>(
            signal_ptrs[index]);
    buffers.pointers[index] = reinterpret_cast<__nv_bfloat16*>(
        buffer_ptrs[index]);
  }
  int const gate_type = shared_gate.scalar_type() == at::kFloat
      ? 1
      : (shared_gate.scalar_type() == at::kHalf ? 2 : 3);
  int const gate_stride = shared_gate.numel() == tokens
      ? 1
      : static_cast<int>(shared_gate.size(1));
  int const gate_column =
      shared_gate.numel() == tokens ? 0 : gate_stride - 1;
  int64_t const total_packs = tokens * hidden_size / 8;
  int const blocks = static_cast<int>(std::min<int64_t>(
      tp2_fused_reduce::kMaxBlocks,
      (total_packs + tp2_fused_reduce::kThreads - 1) /
          tp2_fused_reduce::kThreads));
  auto stream = c10::cuda::getCurrentCUDAStream(
      routed_output.get_device());
  tp2_fused_reduce::kernel
      <<<blocks, tp2_fused_reduce::kThreads, 0, stream.stream()>>>(
          output.data_ptr<at::BFloat16>(),
          routed_output.data_ptr<at::BFloat16>(),
          topk_weights.data_ptr<float>(),
          inverse_permutation.data_ptr<int32_t>(),
          shared_output.data_ptr<at::BFloat16>(),
          shared_gate.data_ptr(),
          gate_type,
          gate_stride,
          gate_column,
          static_cast<int>(tokens),
          8,
          static_cast<int>(hidden_size),
          static_cast<int>(routed_output.size(0)),
          signals,
          buffers,
          static_cast<int>(rank));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen35_w1_splitk_silu_mxfp8_out_cuda(
    at::Tensor& output,
    at::Tensor& output_scales,
    at::Tensor& partial,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    bool include_shared,
    std::optional<at::Tensor> shared_weight_arg,
    std::optional<at::Tensor> shared_weight_scales_arg) {
  TORCH_CHECK(
      activation.is_cuda() &&
          (activation.scalar_type() == at::kByte ||
          activation.scalar_type() == at::kFloat8_e4m3fn) &&
          activation.is_contiguous() &&
          activation.dim() == 2 &&
          (activation.size(0) == 1 ||
           activation.size(0) == 2 ||
           activation.size(0) == 4 ||
           activation.size(0) == 8) &&
          activation.size(1) == 2048,
      "activation must be contiguous CUDA MXFP8 [B,2048], "
      "with B in {1,2,4,8}");
  c10::cuda::CUDAGuard device_guard(activation.device());
  auto const device = activation.device();
  int const tokens = static_cast<int>(activation.size(0));
  TORCH_CHECK(
      shared_weight_arg.has_value() ==
          shared_weight_scales_arg.has_value(),
      "shared_weight and shared_weight_scales must be provided together");
  bool const separate_shared = shared_weight_arg.has_value();
  TORCH_CHECK(
      !separate_shared ||
          (include_shared &&
           (tokens == 1 || tokens == 2 || tokens == 4)),
      "separate shared W1 currently requires include_shared=True and "
      "B in {1,2,4}");
  int const experts = include_shared && !separate_shared ? 257 : 256;
  int const routes_per_token = include_shared ? 9 : 8;
  int const routes = tokens * routes_per_token;
  TORCH_CHECK(
      logical_scales.is_cuda() &&
          logical_scales.device() == device &&
          logical_scales.scalar_type() == at::kByte &&
          logical_scales.is_contiguous() &&
          logical_scales.sizes() ==
              at::IntArrayRef({tokens, 64}),
      "logical_scales must be contiguous CUDA uint8 [B,64]");
  TORCH_CHECK(
      weight.is_cuda() &&
          weight.device() == device &&
          weight.scalar_type() == at::kByte &&
          weight.is_contiguous() &&
          weight.dim() == 3 &&
          weight.size(0) == experts &&
          weight.size(1) == 512 &&
          weight.size(2) == 1536,
      "weight must be canonical packed MXFP6 [",
      experts,
      ",512,1536]");
  TORCH_CHECK(
      weight_scales.is_cuda() &&
          weight_scales.device() == device &&
          weight_scales.scalar_type() == at::kByte &&
          weight_scales.is_contiguous() &&
          weight_scales.numel() >=
              static_cast<int64_t>(experts) * 512 * 64,
      "weight_scales is too small for [",
      experts,
      ",512,64]");
  if (separate_shared) {
    at::Tensor const& shared_weight = *shared_weight_arg;
    at::Tensor const& shared_weight_scales =
        *shared_weight_scales_arg;
    TORCH_CHECK(
        shared_weight.is_cuda() &&
            shared_weight.device() == device &&
            shared_weight.scalar_type() == at::kByte &&
            shared_weight.is_contiguous() &&
            shared_weight.sizes() == at::IntArrayRef({1, 512, 1536}),
        "shared_weight must be canonical packed MXFP6 [1,512,1536]");
    TORCH_CHECK(
        shared_weight_scales.is_cuda() &&
            shared_weight_scales.device() == device &&
            shared_weight_scales.scalar_type() == at::kByte &&
            shared_weight_scales.is_contiguous() &&
            shared_weight_scales.numel() >= 512 * 64,
        "shared_weight_scales is too small for [1,512,64]");
  }
  TORCH_CHECK(
      topk_ids.is_cuda() &&
          topk_ids.device() == device &&
          topk_ids.scalar_type() == at::kInt &&
          topk_ids.is_contiguous() &&
          topk_ids.sizes() ==
              at::IntArrayRef({tokens, 8}),
      "topk_ids must be contiguous CUDA int32 [B,8]");
  TORCH_CHECK(
      output.is_cuda() &&
          output.device() == device &&
          output.scalar_type() == at::kByte &&
          output.is_contiguous() &&
          output.sizes() ==
              at::IntArrayRef({routes, 256}),
      "output must be contiguous CUDA uint8 [B*",
      routes_per_token,
      ",256]");
  TORCH_CHECK(
      output_scales.is_cuda() &&
          output_scales.device() == device &&
          output_scales.scalar_type() == at::kByte &&
          output_scales.is_contiguous() &&
          output_scales.sizes() ==
              at::IntArrayRef({routes, 8}),
      "output_scales must be contiguous CUDA uint8 [B*",
      routes_per_token,
      ",8]");
  TORCH_CHECK(
      partial.is_cuda() &&
          partial.device() == device &&
          partial.scalar_type() == at::kFloat &&
          partial.is_contiguous() &&
          partial.numel() >=
              (include_shared
                   ? ((tokens == 4 || tokens == 8)
                          ? 288 * 128
                          : 144 * 128)
                   : ((tokens == 4 || tokens == 8)
                          ? 256 * 128
                          : 128 * 128)),
      "partial must be contiguous CUDA float32 with enough split-K storage");

  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(
          activation.get_device());
  TORCH_CHECK(
      properties.major == 12 &&
          properties.minor == 0 &&
          properties.cooperativeLaunch,
      "Qwen3.5 split-K W1 requires cooperative SM120");
  int active_blocks = 0;
  bool use_compact_four_token_grid = false;
  if (separate_shared && tokens == 1) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<1, 4, true, true>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else if (separate_shared && tokens == 2) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<2, 2, true, true>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else if (separate_shared) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<4, 2, true, true>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else if (include_shared && tokens == 1) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<1>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else if (include_shared && tokens == 2) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<2>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else if (include_shared && tokens == 4) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<4, 2>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else if (include_shared) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<8, 1>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else if (tokens == 1) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<1, 4, false>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else if (tokens == 2) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<2, 2, false>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else if (tokens == 4) {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<4, 2, false>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  } else {
    C10_CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            direct_moe::qwen35_w1_splitk_kernel<8, 1, false>,
            direct_moe::kThreads,
            sizeof(direct_moe::SplitKSharedStorage)));
  }
  int grid_blocks =
      include_shared
      ? (tokens == 4 || tokens == 8 ? 288 : 144)
      : (tokens == 4 || tokens == 8 ? 256 : 128);
  // The fused B=4/8 path normally uses 288 cooperative CTAs.  Smaller SM120
  // parts cannot make that entire grid resident even though the underlying
  // work fits comfortably.  Preserve the fused schedule by using a single-K
  // split B=4 grid (144 CTAs); B=8 is issued as two such grids with disjoint
  // partial/output storage.  Programmatic dependency lets the second grid
  // overlap the first grid's short reduction tail without violating the
  // cooperative-grid residency requirement.
  if ((tokens == 4 || tokens == 8) &&
      (!separate_shared || tokens == 4) &&
      active_blocks * properties.multiProcessorCount < grid_blocks) {
    if (separate_shared) {
      C10_CUDA_CHECK(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_blocks,
              direct_moe::qwen35_w1_splitk_kernel<4, 1, true, true>,
              direct_moe::kThreads,
              sizeof(direct_moe::SplitKSharedStorage)));
    } else if (include_shared) {
      C10_CUDA_CHECK(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_blocks,
              direct_moe::qwen35_w1_splitk_kernel<4, 1>,
              direct_moe::kThreads,
              sizeof(direct_moe::SplitKSharedStorage)));
    } else {
      C10_CUDA_CHECK(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_blocks,
              direct_moe::qwen35_w1_splitk_kernel<4, 1, false>,
              direct_moe::kThreads,
              sizeof(direct_moe::SplitKSharedStorage)));
    }
    use_compact_four_token_grid = true;
    grid_blocks = include_shared ? 144 : 128;
  }
  TORCH_CHECK(
      active_blocks * properties.multiProcessorCount >= grid_blocks,
      "Qwen3.5 split-K W1 needs ",
      grid_blocks,
      " resident CTAs; device supports ",
      active_blocks * properties.multiProcessorCount);

  auto const* activation_ptr =
      static_cast<uint8_t const*>(activation.data_ptr());
  auto const* scales_ptr =
      logical_scales.data_ptr<uint8_t>();
  auto const* weight_ptr =
      weight.data_ptr<uint8_t>();
  auto const* weight_scales_ptr =
      weight_scales.data_ptr<uint8_t>();
  uint8_t const* shared_weight_ptr = separate_shared
      ? shared_weight_arg->data_ptr<uint8_t>()
      : nullptr;
  uint8_t const* shared_weight_scales_ptr = separate_shared
      ? shared_weight_scales_arg->data_ptr<uint8_t>()
      : nullptr;
  auto const* ids_ptr = topk_ids.data_ptr<int32_t>();
  auto* partial_ptr = partial.data_ptr<float>();
  auto* output_ptr = output.data_ptr<uint8_t>();
  auto* output_scales_ptr =
      output_scales.data_ptr<uint8_t>();
  auto stream = c10::cuda::getCurrentCUDAStream(
      activation.get_device());
  bool const use_b4_cluster_reduce =
      include_shared && !separate_shared && tokens == 4 &&
      !use_compact_four_token_grid;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid_blocks);
  config.blockDim = dim3(direct_moe::kThreads);
  config.dynamicSmemBytes =
      sizeof(direct_moe::SplitKSharedStorage);
  config.stream = stream.stream();
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id =
      cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val
      .programmaticStreamSerializationAllowed = 1;
  if (use_b4_cluster_reduce) {
    attributes[1].id = cudaLaunchAttributeClusterDimension;
    attributes[1].val.clusterDim.x = 2;
    attributes[1].val.clusterDim.y = 1;
    attributes[1].val.clusterDim.z = 1;
  } else {
    attributes[1].id = cudaLaunchAttributeCooperative;
    attributes[1].val.cooperative = 1;
  }
  config.attrs = attributes;
  config.numAttrs = 2;
  if (use_b4_cluster_reduce) {
    int active_clusters = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(
        &active_clusters,
        direct_moe::qwen35_w1_splitk_kernel<
            4, 2, true, false, true>,
        &config));
    TORCH_CHECK(
        active_clusters > 0,
        "Qwen3.5 B4 cluster-reduce W1 has zero active clusters");
  }
  if (separate_shared && tokens == 1) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<1, 4, true, true>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        shared_weight_ptr,
        shared_weight_scales_ptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (separate_shared && tokens == 2) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<2, 2, true, true>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        shared_weight_ptr,
        shared_weight_scales_ptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (separate_shared && use_compact_four_token_grid) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<4, 1, true, true>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        shared_weight_ptr,
        shared_weight_scales_ptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (separate_shared) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<4, 2, true, true>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        shared_weight_ptr,
        shared_weight_scales_ptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (include_shared && tokens == 1) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<1>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (include_shared && tokens == 2) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<2>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (include_shared && tokens == 4 &&
             use_compact_four_token_grid) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<4, 1>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (include_shared && tokens == 8 &&
             use_compact_four_token_grid) {
    constexpr int kTokenOffset = 4;
    constexpr int kRouteOffset = kTokenOffset * 9;
    constexpr int kPartialBlocks = 144;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<4, 1>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<4, 1>,
        activation_ptr + kTokenOffset * 2048,
        scales_ptr + kTokenOffset * 64,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr + kTokenOffset * 8,
        partial_ptr + kPartialBlocks * direct_moe::kTileM,
        output_ptr + kRouteOffset * 256,
        output_scales_ptr + kRouteOffset * 8));
  } else if (include_shared && tokens == 4) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<
            4, 2, true, false, true>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (include_shared) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<8, 1>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (tokens == 4 && use_compact_four_token_grid) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<4, 1, false>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (tokens == 8 && use_compact_four_token_grid) {
    constexpr int kTokenOffset = 4;
    constexpr int kRouteOffset = kTokenOffset * 8;
    constexpr int kPartialBlocks = 128;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<4, 1, false>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<4, 1, false>,
        activation_ptr + kTokenOffset * 2048,
        scales_ptr + kTokenOffset * 64,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr + kTokenOffset * 8,
        partial_ptr + kPartialBlocks * direct_moe::kTileM,
        output_ptr + kRouteOffset * 256,
        output_scales_ptr + kRouteOffset * 8));
  } else if (tokens == 1) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<1, 4, false>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (tokens == 2) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<2, 2, false>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else if (tokens == 4) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<4, 2, false>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  } else {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w1_splitk_kernel<8, 1, false>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        partial_ptr,
        output_ptr,
        output_scales_ptr));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen35_w2_splitk_reduce_out_cuda(
    at::Tensor& output,
    at::Tensor& partial,
    at::Tensor const& activation,
    at::Tensor const& logical_scales,
    at::Tensor const& weight,
    at::Tensor const& weight_scales,
    at::Tensor const& topk_ids,
    at::Tensor const& topk_weights,
    at::Tensor const& shared_gate,
    std::optional<at::Tensor> shared_weight_arg,
    std::optional<at::Tensor> shared_weight_scales_arg) {
  TORCH_CHECK(
      activation.is_cuda() &&
          (activation.scalar_type() == at::kByte ||
           activation.scalar_type() == at::kFloat8_e4m3fn) &&
          activation.is_contiguous() &&
          activation.sizes() ==
              at::IntArrayRef({9, 256}),
      "activation must be contiguous CUDA MXFP8 [9,256]");
  c10::cuda::CUDAGuard device_guard(activation.device());
  auto const device = activation.device();
  TORCH_CHECK(
      logical_scales.is_cuda() &&
          logical_scales.device() == device &&
          logical_scales.scalar_type() == at::kByte &&
          logical_scales.is_contiguous() &&
          logical_scales.sizes() ==
              at::IntArrayRef({9, 8}),
      "logical_scales must be contiguous CUDA uint8 [9,8]");
  TORCH_CHECK(
      shared_weight_arg.has_value() ==
          shared_weight_scales_arg.has_value(),
      "shared_weight and shared_weight_scales must be provided together");
  bool const separate_shared = shared_weight_arg.has_value();
  int const experts = separate_shared ? 256 : 257;
  TORCH_CHECK(
      weight.is_cuda() &&
          weight.device() == device &&
          weight.scalar_type() == at::kByte &&
          weight.is_contiguous() &&
          weight.dim() == 3 &&
          weight.size(0) == experts &&
          weight.size(1) == 2048 &&
          weight.size(2) == 192,
      "weight must be canonical packed MXFP6 [",
      experts,
      ",2048,192]");
  TORCH_CHECK(
      weight_scales.is_cuda() &&
          weight_scales.device() == device &&
          weight_scales.scalar_type() == at::kByte &&
          weight_scales.is_contiguous() &&
          weight_scales.numel() >=
              static_cast<int64_t>(experts) * 2048 * 8,
      "weight_scales is too small for [",
      experts,
      ",2048,8]");
  if (separate_shared) {
    at::Tensor const& shared_weight = *shared_weight_arg;
    at::Tensor const& shared_weight_scales =
        *shared_weight_scales_arg;
    TORCH_CHECK(
        shared_weight.is_cuda() &&
            shared_weight.device() == device &&
            shared_weight.scalar_type() == at::kByte &&
            shared_weight.is_contiguous() &&
            shared_weight.sizes() == at::IntArrayRef({1, 2048, 192}),
        "shared_weight must be canonical packed MXFP6 [1,2048,192]");
    TORCH_CHECK(
        shared_weight_scales.is_cuda() &&
            shared_weight_scales.device() == device &&
            shared_weight_scales.scalar_type() == at::kByte &&
            shared_weight_scales.is_contiguous() &&
            shared_weight_scales.numel() >= 2048 * 8,
        "shared_weight_scales is too small for [1,2048,8]");
  }
  TORCH_CHECK(
      topk_ids.is_cuda() &&
          topk_ids.device() == device &&
          topk_ids.scalar_type() == at::kInt &&
          topk_ids.is_contiguous() &&
          topk_ids.sizes() ==
              at::IntArrayRef({1, 8}),
      "topk_ids must be contiguous CUDA int32 [1,8]");
  TORCH_CHECK(
      topk_weights.is_cuda() &&
          topk_weights.device() == device &&
          topk_weights.scalar_type() == at::kFloat &&
          topk_weights.is_contiguous() &&
          topk_weights.sizes() ==
              at::IntArrayRef({1, 8}),
      "topk_weights must be contiguous CUDA float32 [1,8]");
  TORCH_CHECK(
      shared_gate.is_cuda() &&
          shared_gate.device() == device &&
          shared_gate.scalar_type() == at::kBFloat16 &&
          shared_gate.is_contiguous() &&
          shared_gate.numel() == 1,
      "shared_gate must be contiguous CUDA bfloat16 [1]");
  TORCH_CHECK(
      output.is_cuda() &&
          output.device() == device &&
          output.scalar_type() == at::kBFloat16 &&
          output.is_contiguous() &&
          output.sizes() ==
              at::IntArrayRef({1, 2048}),
      "output must be contiguous CUDA bfloat16 [1,2048]");
  TORCH_CHECK(
      partial.is_cuda() &&
          partial.device() == device &&
          partial.scalar_type() == at::kFloat &&
          partial.is_contiguous() &&
          partial.sizes() ==
              at::IntArrayRef({256, 9, 16}),
      "partial must be contiguous CUDA float32 [256,9,16]");

  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(
          activation.get_device());
  TORCH_CHECK(
      properties.major == 12 &&
          properties.minor == 0 &&
          properties.cooperativeLaunch,
      "Qwen3.5 split-K W2 requires cooperative SM120");
  int active_blocks = 0;
  C10_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          separate_shared
              ? direct_moe::qwen35_w2_splitk_reduce_kernel<true>
              : direct_moe::qwen35_w2_splitk_reduce_kernel<false>,
          direct_moe::kReduceThreads,
          sizeof(direct_moe::ReduceSharedStorage)));
  TORCH_CHECK(
      active_blocks * properties.multiProcessorCount >= 256,
      "Qwen3.5 split-K W2 needs 256 resident CTAs; device supports ",
      active_blocks * properties.multiProcessorCount);

  auto const* activation_ptr =
      static_cast<uint8_t const*>(activation.data_ptr());
  auto const* scales_ptr =
      logical_scales.data_ptr<uint8_t>();
  auto const* weight_ptr =
      weight.data_ptr<uint8_t>();
  auto const* weight_scales_ptr =
      weight_scales.data_ptr<uint8_t>();
  uint8_t const* shared_weight_ptr = separate_shared
      ? shared_weight_arg->data_ptr<uint8_t>()
      : nullptr;
  uint8_t const* shared_weight_scales_ptr = separate_shared
      ? shared_weight_scales_arg->data_ptr<uint8_t>()
      : nullptr;
  auto const* ids_ptr = topk_ids.data_ptr<int32_t>();
  auto const* topk_weights_ptr =
      topk_weights.data_ptr<float>();
  auto const* shared_gate_ptr =
      reinterpret_cast<cutlass::bfloat16_t const*>(
          shared_gate.data_ptr());
  auto* partial_ptr = partial.data_ptr<float>();
  auto* output_ptr =
      reinterpret_cast<cutlass::bfloat16_t*>(
          output.data_ptr());
  auto stream = c10::cuda::getCurrentCUDAStream(
      activation.get_device());
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(256);
  config.blockDim = dim3(direct_moe::kReduceThreads);
  config.dynamicSmemBytes =
      sizeof(direct_moe::ReduceSharedStorage);
  config.stream = stream.stream();
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id =
      cudaLaunchAttributeCooperative;
  attributes[0].val.cooperative = 1;
  attributes[1].id =
      cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[1].val
      .programmaticStreamSerializationAllowed = 1;
  config.attrs = attributes;
  config.numAttrs = 2;
  if (separate_shared) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w2_splitk_reduce_kernel<true>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        shared_weight_ptr,
        shared_weight_scales_ptr,
        ids_ptr,
        topk_weights_ptr,
        shared_gate_ptr,
        partial_ptr,
        output_ptr));
  } else {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        direct_moe::qwen35_w2_splitk_reduce_kernel<false>,
        activation_ptr,
        scales_ptr,
        weight_ptr,
        weight_scales_ptr,
        nullptr,
        nullptr,
        ids_ptr,
        topk_weights_ptr,
        shared_gate_ptr,
        partial_ptr,
        output_ptr));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen35_router_quant_out_cuda(
    at::Tensor& quantized,
    at::Tensor& logical_scales,
    at::Tensor& routed_logits,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    at::Tensor& shared_gate,
    at::Tensor const& hidden,
    at::Tensor const& gate_weight,
    bool renormalize,
    std::optional<at::Tensor> shared_gate_weight_arg) {
  TORCH_CHECK(
      hidden.is_cuda() &&
          hidden.scalar_type() == at::kBFloat16 &&
          hidden.is_contiguous() &&
          hidden.dim() == 2,
      "hidden must be contiguous CUDA bfloat16 [tokens,2048]");
  c10::cuda::CUDAGuard device_guard(hidden.device());
  auto const device = hidden.device();
  int64_t const tokens = hidden.size(0);
  TORCH_CHECK(
      tokens == 1 || tokens == 2 || tokens == 4 || tokens == 8 ||
          tokens == 16,
      "Qwen3.5 fused router supports token counts 1, 2, 4, 8, or 16; got ",
      tokens);
  TORCH_CHECK(
      hidden.size(1) == qwen35_router::kHidden,
      "Qwen3.5 fused router requires hidden size 2048");
  bool const separate_shared = shared_gate_weight_arg.has_value();
  int const gate_rows = separate_shared
      ? qwen35_router::kRoutedExperts
      : qwen35_router::kExpertsWithSharedGate;
  TORCH_CHECK(
      gate_weight.is_cuda() &&
          gate_weight.device() == device &&
          gate_weight.scalar_type() == at::kBFloat16 &&
          gate_weight.is_contiguous() &&
          gate_weight.dim() == 2 &&
          gate_weight.size(0) == gate_rows &&
          gate_weight.size(1) == qwen35_router::kHidden,
      "gate_weight must be contiguous CUDA bfloat16 [",
      gate_rows,
      ",2048]");
  if (separate_shared) {
    at::Tensor const& shared_gate_weight = *shared_gate_weight_arg;
    TORCH_CHECK(
        shared_gate_weight.is_cuda() &&
            shared_gate_weight.device() == device &&
            shared_gate_weight.scalar_type() == at::kBFloat16 &&
            shared_gate_weight.is_contiguous() &&
            shared_gate_weight.sizes() ==
                at::IntArrayRef({1, qwen35_router::kHidden}),
        "shared_gate_weight must be contiguous CUDA bfloat16 [1,2048]");
  }
  TORCH_CHECK(
      quantized.is_cuda() &&
          quantized.device() == device &&
          quantized.scalar_type() == at::kByte &&
          quantized.is_contiguous() &&
          quantized.sizes() == hidden.sizes(),
      "quantized must be contiguous CUDA uint8 [tokens,2048]");
  TORCH_CHECK(
      logical_scales.is_cuda() &&
          logical_scales.device() == device &&
          logical_scales.scalar_type() == at::kByte &&
          logical_scales.is_contiguous() &&
          logical_scales.dim() == 2 &&
          logical_scales.size(0) == tokens &&
          logical_scales.size(1) ==
              qwen35_router::kGroupsPerRow,
      "logical_scales must be contiguous CUDA uint8 [tokens,64]");
  TORCH_CHECK(
      routed_logits.is_cuda() &&
          routed_logits.device() == device &&
          routed_logits.scalar_type() == at::kBFloat16 &&
          routed_logits.is_contiguous() &&
          routed_logits.dim() == 2 &&
          routed_logits.size(0) == tokens &&
          routed_logits.size(1) ==
              qwen35_router::kRoutedExperts,
      "routed_logits must be contiguous CUDA bfloat16 [tokens,256]");
  TORCH_CHECK(
      topk_weights.is_cuda() &&
          topk_weights.device() == device &&
          topk_weights.scalar_type() == at::kFloat &&
          topk_weights.is_contiguous() &&
          topk_weights.dim() == 2 &&
          topk_weights.size(0) == tokens &&
          topk_weights.size(1) == qwen35_router::kTopK,
      "topk_weights must be contiguous CUDA float32 [tokens,8]");
  TORCH_CHECK(
      topk_ids.is_cuda() &&
          topk_ids.device() == device &&
          topk_ids.scalar_type() == at::kInt &&
          topk_ids.is_contiguous() &&
          topk_ids.sizes() == topk_weights.sizes(),
      "topk_ids must be contiguous CUDA int32 [tokens,8]");
  TORCH_CHECK(
      shared_gate.is_cuda() &&
          shared_gate.device() == device &&
          shared_gate.scalar_type() == at::kBFloat16 &&
          shared_gate.is_contiguous() &&
          shared_gate.numel() == tokens,
      "shared_gate must be contiguous CUDA bfloat16 [tokens]");
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(hidden.get_device());
  TORCH_CHECK(
      properties.major == 12 && properties.minor == 0,
      "Qwen3.5 fused router requires SM120");
  TORCH_CHECK(
      properties.cooperativeLaunch,
      "Qwen3.5 fused router requires cooperative kernel launch");
  auto stream =
      c10::cuda::getCurrentCUDAStream(hidden.get_device());
  at::Tensor const* shared_gate_weight = separate_shared
      ? &*shared_gate_weight_arg
      : nullptr;
  if (separate_shared && tokens == 1) {
    qwen35_router::launch<1, true>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, shared_gate_weight,
        renormalize, stream.stream());
  } else if (separate_shared && tokens == 2) {
    qwen35_router::launch<2, true>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, shared_gate_weight,
        renormalize, stream.stream());
  } else if (separate_shared && tokens == 4) {
    qwen35_router::launch<4, true>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, shared_gate_weight,
        renormalize, stream.stream());
  } else if (separate_shared && tokens == 8) {
    qwen35_router::launch<8, true>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, shared_gate_weight,
        renormalize, stream.stream());
  } else if (separate_shared) {
    qwen35_router::launch<16, true>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, shared_gate_weight,
        renormalize, stream.stream());
  } else if (tokens == 1) {
    qwen35_router::launch<1>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, nullptr, renormalize, stream.stream());
  } else if (tokens == 2) {
    qwen35_router::launch<2>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, nullptr, renormalize, stream.stream());
  } else if (tokens == 4) {
    qwen35_router::launch<4>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, nullptr, renormalize, stream.stream());
  } else if (tokens == 8) {
    qwen35_router::launch<8>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, nullptr, renormalize, stream.stream());
  } else {
    qwen35_router::launch<16>(
        quantized, logical_scales, routed_logits,
        topk_weights, topk_ids, shared_gate,
        hidden, gate_weight, nullptr, renormalize, stream.stream());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen35_topk_quant_out_cuda(
    at::Tensor& quantized,
    at::Tensor& logical_scales,
    at::Tensor& packed_scales,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    at::Tensor const& hidden,
    at::Tensor const& routed_logits,
    bool renormalize) {
  TORCH_CHECK(
      hidden.is_cuda() &&
          hidden.scalar_type() == at::kBFloat16 &&
          hidden.is_contiguous() &&
          hidden.dim() == 2 &&
          hidden.size(0) > 0 &&
          hidden.size(0) <= std::numeric_limits<int>::max() &&
          hidden.size(1) == qwen35_router::kHidden,
      "hidden must be contiguous CUDA bfloat16 [B,2048] and B must fit "
      "int32");
  c10::cuda::CUDAGuard device_guard(hidden.device());
  auto const device = hidden.device();
  int64_t const tokens = hidden.size(0);
  TORCH_CHECK(
      routed_logits.is_cuda() &&
          routed_logits.device() == device &&
          routed_logits.scalar_type() == at::kBFloat16 &&
          routed_logits.is_contiguous() &&
          routed_logits.dim() == 2 &&
          routed_logits.size(0) == tokens &&
          routed_logits.size(1) ==
              qwen35_router::kRoutedExperts,
      "routed_logits must be contiguous CUDA bfloat16 [B,256]");
  TORCH_CHECK(
      quantized.is_cuda() &&
          quantized.device() == device &&
          quantized.scalar_type() == at::kByte &&
          quantized.is_contiguous() &&
          quantized.sizes() == hidden.sizes(),
      "quantized must be contiguous CUDA uint8 [B,2048]");
  TORCH_CHECK(
      logical_scales.is_cuda() &&
          logical_scales.device() == device &&
          logical_scales.scalar_type() == at::kByte &&
          logical_scales.is_contiguous() &&
          logical_scales.dim() == 2 &&
          logical_scales.size(0) == tokens &&
          logical_scales.size(1) ==
              qwen35_router::kGroupsPerRow,
      "logical_scales must be contiguous CUDA uint8 [B,64]");
  int64_t const padded_rows =
      (tokens + kScaleRowsPerAtom - 1) /
      kScaleRowsPerAtom * kScaleRowsPerAtom;
  TORCH_CHECK(
      packed_scales.is_cuda() &&
          packed_scales.device() == device &&
          packed_scales.scalar_type() == at::kByte &&
          packed_scales.is_contiguous() &&
          packed_scales.numel() >=
              padded_rows * qwen35_router::kGroupsPerRow,
      "packed_scales is too small for the SM120 scale layout");
  TORCH_CHECK(
      topk_weights.is_cuda() &&
          topk_weights.device() == device &&
          topk_weights.scalar_type() == at::kFloat &&
          topk_weights.is_contiguous() &&
          topk_weights.dim() == 2 &&
          topk_weights.size(0) == tokens &&
          topk_weights.size(1) == qwen35_router::kTopK,
      "topk_weights must be contiguous CUDA float32 [B,8]");
  TORCH_CHECK(
      topk_ids.is_cuda() &&
          topk_ids.device() == device &&
          topk_ids.scalar_type() == at::kInt &&
          topk_ids.is_contiguous() &&
          topk_ids.sizes() == topk_weights.sizes(),
      "topk_ids must be contiguous CUDA int32 [B,8]");
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(hidden.get_device());
  TORCH_CHECK(
      properties.major == 12 && properties.minor == 0,
      "Qwen3.5 fused top-k quantization requires SM120");
  auto stream =
      c10::cuda::getCurrentCUDAStream(hidden.get_device());
  qwen35_router::topk_quant_kernel<false>
      <<<static_cast<int>(tokens),
         qwen35_router::kThreads,
         0,
         stream.stream()>>>(
          reinterpret_cast<__nv_bfloat16 const*>(
              hidden.data_ptr()),
          reinterpret_cast<__nv_bfloat16 const*>(
              routed_logits.data_ptr()),
          quantized.data_ptr<uint8_t>(),
          logical_scales.data_ptr<uint8_t>(),
          packed_scales.data_ptr<uint8_t>(),
          topk_weights.data_ptr<float>(),
          topk_ids.data_ptr<int32_t>(),
          static_cast<int>(tokens),
          renormalize,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          {});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qwen35_topk_quant_route_out_cuda(
    at::Tensor& quantized,
    at::Tensor& logical_scales,
    at::Tensor& dense_scales,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    at::Tensor& permuted_activation,
    at::Tensor& grouped_scales,
    at::Tensor& expert_cursors,
    at::Tensor& expert_offsets,
    at::Tensor& scale_offsets,
    at::Tensor& inverse_permutation,
    at::Tensor const& hidden,
    at::Tensor const& routed_logits,
    std::optional<at::Tensor> grouped_output_arg,
    std::optional<at::Tensor> weight_arg,
    std::optional<at::Tensor> weight_scales_arg,
    std::optional<at::Tensor> source_tokens_arg,
    bool renormalize) {
  TORCH_CHECK(
      hidden.is_cuda() &&
          hidden.scalar_type() == at::kBFloat16 &&
          hidden.is_contiguous() &&
          hidden.dim() == 2 &&
          hidden.size(0) > 0 &&
          hidden.size(0) <= 256 &&
          hidden.size(1) == qwen35_router::kHidden,
      "hidden must be contiguous CUDA bfloat16 [B,2048], 1 <= B <= 256");
  c10::cuda::CUDAGuard device_guard(hidden.device());
  auto const device = hidden.device();
  int64_t const tokens = hidden.size(0);
  int64_t const routes =
      tokens * qwen35_router::kTopK;
  TORCH_CHECK(
      routed_logits.is_cuda() &&
          routed_logits.device() == device &&
          routed_logits.scalar_type() == at::kBFloat16 &&
          routed_logits.is_contiguous() &&
          routed_logits.sizes() ==
              at::IntArrayRef(
                  {tokens, qwen35_router::kRoutedExperts}),
      "routed_logits must be contiguous CUDA bfloat16 [B,256]");
  TORCH_CHECK(
      quantized.is_cuda() &&
          quantized.device() == device &&
          quantized.scalar_type() == at::kByte &&
          quantized.is_contiguous() &&
          quantized.sizes() == hidden.sizes(),
      "quantized must be contiguous CUDA uint8 [B,2048]");
  TORCH_CHECK(
      logical_scales.is_cuda() &&
          logical_scales.device() == device &&
          logical_scales.scalar_type() == at::kByte &&
          logical_scales.is_contiguous() &&
          logical_scales.sizes() ==
              at::IntArrayRef(
                  {tokens, qwen35_router::kGroupsPerRow}),
      "logical_scales must be contiguous CUDA uint8 [B,64]");
  int64_t const dense_scale_rows =
      (tokens + kScaleRowsPerAtom - 1) /
      kScaleRowsPerAtom * kScaleRowsPerAtom;
  TORCH_CHECK(
      dense_scales.is_cuda() &&
          dense_scales.device() == device &&
          dense_scales.scalar_type() == at::kByte &&
          dense_scales.is_contiguous() &&
          dense_scales.numel() >=
              dense_scale_rows *
                  qwen35_router::kGroupsPerRow,
      "dense_scales is too small for the SM120 scale layout");
  TORCH_CHECK(
      topk_weights.is_cuda() &&
          topk_weights.device() == device &&
          topk_weights.scalar_type() == at::kFloat &&
          topk_weights.is_contiguous() &&
          topk_weights.sizes() ==
              at::IntArrayRef(
                  {tokens, qwen35_router::kTopK}),
      "topk_weights must be contiguous CUDA float32 [B,8]");
  TORCH_CHECK(
      topk_ids.is_cuda() &&
          topk_ids.device() == device &&
          topk_ids.scalar_type() == at::kInt &&
          topk_ids.is_contiguous() &&
          topk_ids.sizes() == topk_weights.sizes(),
      "topk_ids must be contiguous CUDA int32 [B,8]");
  TORCH_CHECK(
      permuted_activation.is_cuda() &&
          permuted_activation.device() == device &&
          (permuted_activation.scalar_type() ==
               at::kFloat8_e4m3fn ||
           permuted_activation.scalar_type() == at::kByte) &&
          permuted_activation.is_contiguous() &&
          permuted_activation.sizes() ==
              at::IntArrayRef(
                  {routes, qwen35_router::kHidden}),
      "permuted_activation must be contiguous CUDA MXFP8 [B*8,2048]");
  int64_t const grouped_scale_rows =
      routes +
      std::min<int64_t>(
          routes, qwen35_router::kRoutedExperts) *
          (kScaleRowsPerAtom - 1);
  TORCH_CHECK(
      grouped_scales.is_cuda() &&
          grouped_scales.device() == device &&
          grouped_scales.scalar_type() == at::kByte &&
          grouped_scales.is_contiguous() &&
          grouped_scales.numel() >=
              grouped_scale_rows *
                  qwen35_router::kGroupsPerRow,
      "grouped_scales is too small for routed SM120 scale packing");
  TORCH_CHECK(
      expert_cursors.is_cuda() &&
          expert_cursors.device() == device &&
          expert_cursors.scalar_type() == at::kInt &&
          expert_cursors.is_contiguous() &&
          expert_cursors.numel() ==
              qwen35_router::kRoutedExperts,
      "expert_cursors must be contiguous CUDA int32 [256]");
  TORCH_CHECK(
      expert_offsets.is_cuda() &&
          expert_offsets.device() == device &&
          expert_offsets.scalar_type() == at::kLong &&
          expert_offsets.is_contiguous() &&
          expert_offsets.numel() ==
              qwen35_router::kRoutedExperts + 1,
      "expert_offsets must be contiguous CUDA int64 [257]");
  TORCH_CHECK(
      scale_offsets.is_cuda() &&
          scale_offsets.device() == device &&
          scale_offsets.scalar_type() == at::kLong &&
          scale_offsets.is_contiguous() &&
          scale_offsets.sizes() == expert_offsets.sizes(),
      "scale_offsets must be contiguous CUDA int64 [257]");
  TORCH_CHECK(
      inverse_permutation.is_cuda() &&
          inverse_permutation.device() == device &&
          inverse_permutation.scalar_type() == at::kInt &&
          inverse_permutation.is_contiguous() &&
          inverse_permutation.numel() == routes,
      "inverse_permutation must be contiguous CUDA int32 [B*8]");
  int32_t* source_tokens_ptr = nullptr;
  if (source_tokens_arg.has_value()) {
    auto& source_tokens = *source_tokens_arg;
    TORCH_CHECK(
        source_tokens.is_cuda() && source_tokens.device() == device &&
            source_tokens.scalar_type() == at::kInt &&
            source_tokens.is_contiguous() && source_tokens.dim() == 1 &&
            source_tokens.numel() == routes,
        "source_tokens must be contiguous CUDA int32 [B*8]");
    source_tokens_ptr = source_tokens.data_ptr<int32_t>();
  }
  bool const fused_grouped = grouped_output_arg.has_value();
  TORCH_CHECK(
      source_tokens_ptr == nullptr || !fused_grouped,
      "source_tokens cannot be combined with fused grouped metadata");
  TORCH_CHECK(
      fused_grouped == weight_arg.has_value() &&
          fused_grouped == weight_scales_arg.has_value(),
      "grouped_output, weight, and weight_scales must be provided together");
  std::optional<Qwen35GroupedMetadataStorage> grouped_metadata;
  Qwen35GroupedMetadataDevice grouped_metadata_view{};
  if (fused_grouped) {
    auto& grouped_output = *grouped_output_arg;
    auto const& weight = *weight_arg;
    auto const& weight_scales = *weight_scales_arg;
    int64_t const packed_k = weight.size(2);
    int64_t const gemm_k = packed_k * 4 / 3;
    TORCH_CHECK(
        routes >= qwen35_router::kRoutedExperts &&
            weight.is_cuda() && weight.device() == device &&
            weight.scalar_type() == at::kByte &&
            weight.is_contiguous() && weight.dim() == 3 &&
            weight.size(0) == qwen35_router::kRoutedExperts &&
            packed_k * 4 == gemm_k * 3 &&
            gemm_k == qwen35_router::kHidden,
        "fused grouped W1 requires uint8 MXFP6 [256,N,packed_2048]");
    TORCH_CHECK(
        weight_scales.is_cuda() &&
            weight_scales.device() == device &&
            weight_scales.scalar_type() == at::kByte &&
            weight_scales.is_contiguous(),
        "fused grouped W1 weight_scales must be contiguous CUDA uint8");
    TORCH_CHECK(
        grouped_output.is_cuda() &&
            grouped_output.device() == device &&
            grouped_output.scalar_type() == at::kBFloat16 &&
            grouped_output.is_contiguous() &&
            grouped_output.dim() == 2 &&
            grouped_output.size(0) == routes &&
            grouped_output.size(1) == weight.size(1),
        "fused grouped W1 output must be contiguous CUDA bfloat16 [M,N]");
    grouped_metadata.emplace(
        qwen35_router::kRoutedExperts, permuted_activation);
    grouped_metadata_view = grouped_metadata->device_view(
        permuted_activation,
        grouped_scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
        grouped_output);
  }

  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(hidden.get_device());
  TORCH_CHECK(
      properties.major == 12 &&
          properties.minor == 0 &&
          properties.cooperativeLaunch,
      "Qwen3.5 fused top-k routing requires cooperative SM120");
  int active_blocks = 0;
  C10_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks,
          qwen35_router::topk_quant_kernel<true>,
          qwen35_router::kThreads,
          0));
  TORCH_CHECK(
      tokens <=
          static_cast<int64_t>(active_blocks) *
              properties.multiProcessorCount,
      "Qwen3.5 fused top-k routing needs ",
      tokens,
      " resident CTAs; device supports ",
      active_blocks * properties.multiProcessorCount);

  auto const* hidden_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(
          hidden.data_ptr());
  auto const* logits_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(
          routed_logits.data_ptr());
  auto* quantized_ptr = quantized.data_ptr<uint8_t>();
  auto* logical_ptr =
      logical_scales.data_ptr<uint8_t>();
  auto* dense_ptr = dense_scales.data_ptr<uint8_t>();
  auto* topk_weights_ptr =
      topk_weights.data_ptr<float>();
  auto* topk_ids_ptr = topk_ids.data_ptr<int32_t>();
  auto* permuted_ptr = static_cast<uint8_t*>(
      permuted_activation.data_ptr());
  auto* grouped_ptr =
      grouped_scales.data_ptr<uint8_t>();
  auto* cursors_ptr =
      expert_cursors.data_ptr<int32_t>();
  auto* expert_offsets_ptr =
      expert_offsets.data_ptr<int64_t>();
  auto* scale_offsets_ptr =
      scale_offsets.data_ptr<int64_t>();
  auto* inverse_ptr =
      inverse_permutation.data_ptr<int32_t>();
  int const token_count = static_cast<int>(tokens);
  auto stream =
      c10::cuda::getCurrentCUDAStream(hidden.get_device());
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(token_count);
  config.blockDim = dim3(qwen35_router::kThreads);
  config.stream = stream.stream();
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &config,
      qwen35_router::topk_quant_kernel<true>,
      hidden_ptr,
      logits_ptr,
      quantized_ptr,
      logical_ptr,
      dense_ptr,
      topk_weights_ptr,
      topk_ids_ptr,
      token_count,
      renormalize,
      permuted_ptr,
      grouped_ptr,
      cursors_ptr,
      expert_offsets_ptr,
      scale_offsets_ptr,
      inverse_ptr,
      source_tokens_ptr,
      grouped_metadata_view));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (fused_grouped) {
    launch_qwen35_grouped_from_metadata(
        *grouped_output_arg,
        permuted_activation,
        *grouped_metadata,
        true);
  }
}

}  // namespace mxfp6_gemm::torch_ext

TORCH_LIBRARY_FRAGMENT(mxfp6, m) {
  m.def(
      "quantize_mxfp8_logical(Tensor input) "
      "-> (Tensor values, Tensor scales)");
  m.def(
      "route_mxfp8_out(Tensor activation, Tensor logical_scales, "
      "Tensor topk_ids, Tensor? expert_map, int local_experts, "
      "Tensor(a!) permuted_activation, bool include_shared=False) "
      "-> (Tensor packed_scales, "
      "Tensor expert_offsets, Tensor scale_offsets, "
      "Tensor inverse_permutation)");
  m.def(
      "qwen35_route_mxfp8_out("
      "Tensor activation, Tensor logical_scales, Tensor topk_ids, "
      "Tensor(a!) permuted_activation, Tensor(b!) packed_scales, "
      "Tensor(c!) expert_cursors, Tensor(d!) expert_offsets, "
      "Tensor(e!) scale_offsets, Tensor(f!) inverse_permutation, "
      "bool include_shared=False) -> ()");
  m.def(
      "pack_grouped_scales(Tensor logical, Tensor expert_offsets) "
      "-> (Tensor packed, Tensor scale_offsets)");
  m.def(
      "silu_and_mul_mxfp8_grouped(Tensor input, Tensor expert_offsets, "
      "Tensor? scale_offsets=None) "
      "-> (Tensor values, Tensor scales, Tensor scale_offsets)");
  m.def(
      "silu_and_mul_mxfp8_grouped_out("
      "Tensor(a!) output, Tensor(b!) scales, Tensor input, "
      "Tensor expert_offsets, Tensor scale_offsets, "
      "bool use_pdl=False, Tensor(c!)? grouped_output=None, "
      "Tensor? weight=None, Tensor? weight_scales=None) -> ()");
  m.def(
      "silu_and_mul_mxfp8_logical(Tensor input) "
      "-> (Tensor values, Tensor scales)");
  m.def(
      "silu_and_mul_mxfp8_packed_out("
      "Tensor(a!) output, Tensor(b!) packed_scales, Tensor input) -> ()");
  m.def(
      "array_gemm_w6a8_out(Tensor(a!) output, Tensor activation, "
      "Tensor logical_scales, Tensor weight, Tensor weight_scales, "
      "Tensor topk_ids, bool include_shared=False) -> ()");
  m.def(
      "array_gemm_w6a8_silu_mxfp8_out("
      "Tensor(a!) output, Tensor(b!) output_scales, "
      "Tensor activation, Tensor logical_scales, "
      "Tensor weight, Tensor weight_scales, Tensor topk_ids, "
      "bool include_shared=False) -> ()");
  m.def(
      "array_gemm_w6a8_reduce_out("
      "Tensor(a!) output, Tensor activation, Tensor logical_scales, "
      "Tensor weight, Tensor weight_scales, Tensor topk_ids, "
      "Tensor topk_weights, Tensor shared_gate, "
      "Tensor? shared_output=None, Tensor? shared_weight=None, "
      "Tensor? shared_weight_scales=None, "
      "bool use_packed_vector_loads=False) -> ()");
  m.def(
      "qwen35_router_quant_out("
      "Tensor(a!) quantized, Tensor(b!) logical_scales, "
      "Tensor(c!) routed_logits, Tensor(d!) topk_weights, "
      "Tensor(e!) topk_ids, Tensor(f!) shared_gate, "
      "Tensor hidden, Tensor gate_weight, "
      "bool renormalize=False, "
      "Tensor? shared_gate_weight=None) -> ()");
  m.def(
      "qwen35_topk_quant_out("
      "Tensor(a!) quantized, Tensor(b!) logical_scales, "
      "Tensor(c!) packed_scales, Tensor(d!) topk_weights, "
      "Tensor(e!) topk_ids, Tensor hidden, Tensor routed_logits, "
      "bool renormalize=False) -> ()");
  m.def(
      "qwen35_topk_quant_route_out("
      "Tensor(a!) quantized, Tensor(b!) logical_scales, "
      "Tensor(c!) dense_scales, Tensor(d!) topk_weights, "
      "Tensor(e!) topk_ids, Tensor(f!) permuted_activation, "
      "Tensor(g!) grouped_scales, Tensor(h!) expert_cursors, "
      "Tensor(i!) expert_offsets, Tensor(j!) scale_offsets, "
      "Tensor(k!) inverse_permutation, Tensor hidden, "
      "Tensor routed_logits, Tensor(l!)? grouped_output=None, "
      "Tensor? weight=None, Tensor? weight_scales=None, "
      "Tensor(m!)? source_tokens=None, bool renormalize=False) -> ()");
  m.def(
      "qwen35_w1_splitk_silu_mxfp8_out("
      "Tensor(a!) output, Tensor(b!) output_scales, "
      "Tensor(c!) partial, Tensor activation, "
      "Tensor logical_scales, Tensor weight, "
      "Tensor weight_scales, Tensor topk_ids, "
      "bool include_shared=True, Tensor? shared_weight=None, "
      "Tensor? shared_weight_scales=None) -> ()");
  m.def(
      "qwen35_w2_splitk_reduce_out("
      "Tensor(a!) output, Tensor(b!) partial, "
      "Tensor activation, Tensor logical_scales, "
      "Tensor weight, Tensor weight_scales, "
      "Tensor topk_ids, Tensor topk_weights, "
      "Tensor shared_gate, Tensor? shared_weight=None, "
      "Tensor? shared_weight_scales=None) -> ()");
  m.def(
      "grouped_gemm_w6a8_out(Tensor(a!) output, Tensor activation, "
      "Tensor activation_scales, Tensor weight, Tensor weight_scales, "
      "Tensor expert_offsets, Tensor scale_offsets, int tile_n=0, "
      "bool use_pdl=False) -> ()");
  m.def(
      "qwen35_grouped_w1_silu_mxfp8_out("
      "Tensor(a!) output, Tensor(b!) output_scales, Tensor activation, "
      "Tensor activation_scales, Tensor weight, Tensor weight_scales, "
      "Tensor expert_offsets, Tensor scale_offsets, "
      "bool use_pdl=False, Tensor? source_tokens=None) -> ()");
  m.def(
      "moe_reduce_out(Tensor(a!) output, Tensor routed_output, "
      "Tensor topk_weights, Tensor inverse_permutation, "
      "Tensor? shared_output=None, Tensor? shared_gate=None, "
      "bool use_pdl=False) -> ()");
  m.def(
      "moe_reduce_tp2_out(Tensor(a!) output, Tensor routed_output, "
      "Tensor topk_weights, Tensor inverse_permutation, "
      "Tensor shared_output, Tensor shared_gate, int[] signal_ptrs, "
      "int[] buffer_ptrs, int rank) -> ()");
  m.def(
      "moe_reduce_array_out(Tensor(a!) output, Tensor combined_output, "
      "Tensor topk_weights, Tensor shared_gate) -> ()");
}

TORCH_LIBRARY_IMPL(mxfp6, CUDA, m) {
  m.impl(
      "quantize_mxfp8_logical",
      &mxfp6_gemm::torch_ext::quantize_mxfp8_logical_cuda);
  m.impl(
      "route_mxfp8_out",
      &mxfp6_gemm::torch_ext::route_mxfp8_out_cuda);
  m.impl(
      "qwen35_route_mxfp8_out",
      &mxfp6_gemm::torch_ext::
          qwen35_route_mxfp8_out_cuda);
  m.impl(
      "pack_grouped_scales",
      &mxfp6_gemm::torch_ext::pack_grouped_scales_cuda);
  m.impl(
      "silu_and_mul_mxfp8_grouped",
      &mxfp6_gemm::torch_ext::silu_and_mul_mxfp8_grouped_cuda);
  m.impl(
      "silu_and_mul_mxfp8_grouped_out",
      &mxfp6_gemm::torch_ext::
          silu_and_mul_mxfp8_grouped_out_cuda);
  m.impl(
      "silu_and_mul_mxfp8_logical",
      &mxfp6_gemm::torch_ext::silu_and_mul_mxfp8_logical_cuda);
  m.impl(
      "silu_and_mul_mxfp8_packed_out",
      &mxfp6_gemm::torch_ext::
          silu_and_mul_mxfp8_packed_out_cuda);
  m.impl(
      "array_gemm_w6a8_out",
      &mxfp6_gemm::torch_ext::array_gemm_w6a8_out_cuda);
  m.impl(
      "array_gemm_w6a8_silu_mxfp8_out",
      &mxfp6_gemm::torch_ext::
          array_gemm_w6a8_silu_mxfp8_out_cuda);
  m.impl(
      "array_gemm_w6a8_reduce_out",
      &mxfp6_gemm::torch_ext::
          array_gemm_w6a8_reduce_out_cuda);
  m.impl(
      "qwen35_router_quant_out",
      &mxfp6_gemm::torch_ext::
          qwen35_router_quant_out_cuda);
  m.impl(
      "qwen35_topk_quant_out",
      &mxfp6_gemm::torch_ext::
          qwen35_topk_quant_out_cuda);
  m.impl(
      "qwen35_topk_quant_route_out",
      &mxfp6_gemm::torch_ext::
          qwen35_topk_quant_route_out_cuda);
  m.impl(
      "qwen35_w1_splitk_silu_mxfp8_out",
      &mxfp6_gemm::torch_ext::
          qwen35_w1_splitk_silu_mxfp8_out_cuda);
  m.impl(
      "qwen35_w2_splitk_reduce_out",
      &mxfp6_gemm::torch_ext::
          qwen35_w2_splitk_reduce_out_cuda);
  m.impl(
      "grouped_gemm_w6a8_out",
      &mxfp6_gemm::torch_ext::grouped_gemm_w6a8_out_cuda);
  m.impl(
      "qwen35_grouped_w1_silu_mxfp8_out",
      &mxfp6_gemm::torch_ext::
          qwen35_grouped_w1_silu_mxfp8_out_cuda);
  m.impl(
      "moe_reduce_out",
      &mxfp6_gemm::torch_ext::moe_reduce_out_cuda);
  m.impl(
      "moe_reduce_tp2_out",
      &mxfp6_gemm::torch_ext::moe_reduce_tp2_out_cuda);
  m.impl(
      "moe_reduce_array_out",
      &mxfp6_gemm::torch_ext::moe_reduce_array_out_cuda);
}
