#include "mxfp6_gemm/quantization.hpp"

#include <cstdint>
#include <limits>
#include <tuple>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp6.h>
#include <cuda_fp8.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

namespace mxfp6_gemm::torch_ext {
namespace {

constexpr int kScaleVectorSize = 32;
constexpr int kThreads = 256;
constexpr int kElementsPerThread = 8;
constexpr int kThreadsPerGroup =
    kScaleVectorSize / kElementsPerThread;
constexpr int kGroupsPerBlock = kThreads / kThreadsPerGroup;
constexpr uint8_t kUe8m0One = 0x7f;

int64_t ceil_div(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

int64_t round_up(int64_t value, int64_t alignment) {
  return ceil_div(value, alignment) * alignment;
}

void check_quant_input(at::Tensor const& input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "input must have dtype torch.float16 or torch.bfloat16; got ",
              input.scalar_type());
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must have shape [M,K]");
  int64_t const m = input.size(0);
  int64_t const k = input.size(1);
  TORCH_CHECK(m > 0, "M must be positive; got ", m);
  TORCH_CHECK(k > 0 && k % 128 == 0,
              "K must be a positive multiple of 128; got ", k);
  TORCH_CHECK(m <= std::numeric_limits<int>::max() &&
                  k <= std::numeric_limits<int>::max(),
              "M and K must fit in a 32-bit integer");
  TORCH_CHECK(m <= std::numeric_limits<int64_t>::max() / k,
              "M*K overflows int64");
}

template <int OutputBits>
__device__ __forceinline__ uint16_t quantize_pair(float first,
                                                   float second) {
  if constexpr (OutputBits == 8) {
    return __nv_cvt_float2_to_fp8x2(
        make_float2(first, second), __NV_SATFINITE, __NV_E4M3);
  } else {
    static_assert(OutputBits == 6);
    return __nv_cvt_float2_to_fp6x2(
        make_float2(first, second), __NV_E3M2, cudaRoundNearest);
  }
}

template <class Source, int OutputBits, class ScaleLayout>
__global__ void quantize_mx_kernel(Source const* input,
                                   uint8_t* output,
                                   uint8_t* scales,
                                   int groups_per_row,
                                   int64_t total_groups,
                                   ScaleLayout scale_layout) {
  static_assert(OutputBits == 6 || OutputBits == 8);
  int const thread_in_group = threadIdx.x % kThreadsPerGroup;
  int const group_in_block = threadIdx.x / kThreadsPerGroup;
  int64_t const group =
      static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock + group_in_block;
  bool const valid = group < total_groups;

  float values[kElementsPerThread]{};
  int64_t const value_offset =
      group * kScaleVectorSize + thread_in_group * kElementsPerThread;
  if (valid) {
#pragma unroll
    for (int index = 0; index < kElementsPerThread; ++index) {
      values[index] = static_cast<float>(input[value_offset + index]);
    }
  }

  float absmax = 0.0f;
#pragma unroll
  for (int index = 0; index < kElementsPerThread; ++index) {
    absmax = fmaxf(absmax, fabsf(values[index]));
  }
  absmax = fmaxf(absmax, __shfl_xor_sync(
      0xffffffffu, absmax, 1, kThreadsPerGroup));
  absmax = fmaxf(absmax, __shfl_xor_sync(
      0xffffffffu, absmax, 2, kThreadsPerGroup));

  constexpr float kTargetMax = OutputBits == 8 ? 448.0f : 28.0f;
  float inverse_scale = 1.0f;
  uint8_t scale_code = kUe8m0One;
  if (thread_in_group == 0) {
    float const raw_scale = fmaxf(absmax / kTargetMax, 1.0e-30f);
    uint32_t scale_bits = __float_as_uint(raw_scale);
    // UE8M0 stores only an exponent. Rounding upward guarantees that a finite
    // group maximum cannot overflow the target format after division.
    scale_bits = (scale_bits + 0x007fffffu) & 0x7f800000u;
    scale_code = static_cast<uint8_t>(scale_bits >> 23);
    inverse_scale = 1.0f / __uint_as_float(scale_bits);
  }
  inverse_scale = __shfl_sync(
      0xffffffffu, inverse_scale, 0, kThreadsPerGroup);

  if (valid && thread_in_group == 0) {
    int const row = static_cast<int>(group / groups_per_row);
    int const k_group =
        static_cast<int>(group - static_cast<int64_t>(row) * groups_per_row);
    auto const scale_offset = scale_layout(cute::make_coord(
        row, k_group * kScaleVectorSize, 0));
    scales[scale_offset] = scale_code;
  }

  uint16_t pairs[kElementsPerThread / 2];
#pragma unroll
  for (int index = 0; index < kElementsPerThread; index += 2) {
    pairs[index / 2] = quantize_pair<OutputBits>(
        values[index] * inverse_scale,
        values[index + 1] * inverse_scale);
  }

  if constexpr (OutputBits == 8) {
    if (valid) {
      uint2 packed{
          static_cast<uint32_t>(pairs[0]) |
              (static_cast<uint32_t>(pairs[1]) << 16),
          static_cast<uint32_t>(pairs[2]) |
              (static_cast<uint32_t>(pairs[3]) << 16)};
      reinterpret_cast<uint2*>(output + value_offset)[0] = packed;
    }
  } else {
    uint64_t packed = 0;
#pragma unroll
    for (int index = 0; index < kElementsPerThread / 2; ++index) {
      uint64_t const first = pairs[index] & 0x3fu;
      uint64_t const second = (pairs[index] >> 8) & 0x3fu;
      packed |= first << (index * 12);
      packed |= second << (index * 12 + 6);
    }

    uint32_t const low = static_cast<uint32_t>(packed);
    uint32_t const high = static_cast<uint32_t>(packed >> 32);
    int const partner = (thread_in_group & ~1) + 1;
    // Every lane participates, including invalid groups in the final block.
    // A full-warp shuffle mask is otherwise illegal under subgroup divergence.
    uint32_t const partner_low = __shfl_sync(
        0xffffffffu, low, partner, kThreadsPerGroup);
    uint32_t const partner_high = __shfl_sync(
        0xffffffffu, high, partner, kThreadsPerGroup);
    if (valid && (thread_in_group & 1) == 0) {
      uint8_t* destination = output + group * 24 +
          (thread_in_group / 2) * 12;
      uint32_t* words = reinterpret_cast<uint32_t*>(destination);
      words[0] = low;
      words[1] = (high & 0xffffu) | (partner_low << 16);
      words[2] = (partner_low >> 16) | (partner_high << 16);
    }
  }

  // Let the dependent CUTLASS launch enter residency as soon as this grid has
  // produced all of A and its scales. The CUTLASS consumer performs the
  // matching grid-dependency wait before any global-memory access.
  __syncthreads();
  if (threadIdx.x == 0) {
    cutlass::arch::launch_dependent_grids();
  }
}

template <int OutputBits>
std::tuple<at::Tensor, at::Tensor> quantize_mx(
    at::Tensor const& input) {
  check_quant_input(input);
  c10::cuda::CUDAGuard guard(input.device());
  int64_t const m = input.size(0);
  int64_t const k = input.size(1);
  int64_t const values = m * k;
  int64_t const groups_per_row = k / kScaleVectorSize;
  int64_t const total_groups = m * groups_per_row;
  int64_t const padded_rows = round_up(m, 128);

  auto byte_options = input.options().dtype(at::kByte);
  at::Tensor output = OutputBits == 8
      ? at::empty({m, k}, byte_options)
      : at::empty({values * 3 / 4}, byte_options);
  at::Tensor scales = at::empty(
      {padded_rows * groups_per_row}, byte_options);

  auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      scales.data_ptr<uint8_t>(), kUe8m0One,
      static_cast<size_t>(scales.numel()), stream.stream()));

  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;
  auto const scale_layout = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(static_cast<int>(m), 1, static_cast<int>(k), 1));
  int64_t const block_count = ceil_div(total_groups, kGroupsPerBlock);
  TORCH_CHECK(block_count <= std::numeric_limits<int>::max(),
              "quantization launch grid is too large");

  if (input.scalar_type() == at::kHalf) {
    quantize_mx_kernel<at::Half, OutputBits><<<
        static_cast<int>(block_count), kThreads, 0, stream.stream()>>>(
        input.data_ptr<at::Half>(), output.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(), static_cast<int>(groups_per_row),
        total_groups, scale_layout);
  } else {
    quantize_mx_kernel<at::BFloat16, OutputBits><<<
        static_cast<int>(block_count), kThreads, 0, stream.stream()>>>(
        input.data_ptr<at::BFloat16>(), output.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(), static_cast<int>(groups_per_row),
        total_groups, scale_layout);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> quantize_mxfp8_cuda(
    at::Tensor const& input) {
  return quantize_mx<8>(input);
}

std::tuple<at::Tensor, at::Tensor> quantize_mxfp6_cuda(
    at::Tensor const& input) {
  return quantize_mx<6>(input);
}

}  // namespace mxfp6_gemm::torch_ext
