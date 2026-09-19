#include "mxfp6_gemm/quantization.hpp"

#include <algorithm>
#include <cstdint>
#include <torch/library.h>
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
  TORCH_CHECK(k > 0 && k % kScaleVectorSize == 0,
              "K must be a positive multiple of 32; got ", k);
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

template <class Source, int OutputBits, bool PackedScaleLayout,
          class ScaleLayout, bool SiluAndMul = false>
__global__ void quantize_mx_kernel(Source const* input,
                                   uint8_t* output,
                                   uint8_t* scales,
                                   uint8_t* logical_scales,
                                   int groups_per_row,
                                   int64_t total_groups,
                                   ScaleLayout scale_layout,
                                   bool initialize_padding = false) {
  static_assert(OutputBits == 6 || OutputBits == 8);
  int const thread_in_group = threadIdx.x % kThreadsPerGroup;
  int const group_in_block = threadIdx.x / kThreadsPerGroup;
  int64_t const group =
      static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock + group_in_block;
  bool const valid = group < total_groups;

  // Only padding is initialized here. Logical scale bytes are written below
  // by their owning quantization group, so the two writes never overlap and
  // need no grid synchronization. Replay overwrites every byte, including
  // padding, even when the allocator returns poisoned/stale storage.
  if constexpr (PackedScaleLayout) {
    if (initialize_padding) {
      int64_t const rows = total_groups / groups_per_row;
      int64_t const padded_rows = (rows + 127) / 128 * 128;
      int64_t const packed_groups = (groups_per_row + 3) / 4 * 4;
      for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
           index < padded_rows * packed_groups;
           index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int const row = static_cast<int>(index / packed_groups);
        int const k_group = static_cast<int>(index % packed_groups);
        if (row >= rows || k_group >= groups_per_row) {
          scales[scale_layout(cute::make_coord(row, k_group * kScaleVectorSize, 0))] =
              kUe8m0One;
        }
      }
    }
  }

  float values[kElementsPerThread]{};
  int64_t const value_offset =
      group * kScaleVectorSize + thread_in_group * kElementsPerThread;
  if (valid) {
#pragma unroll
    for (int index = 0; index < kElementsPerThread; ++index) {
      if constexpr (SiluAndMul) {
        int64_t const k = static_cast<int64_t>(groups_per_row) * kScaleVectorSize;
        int64_t const row = group / groups_per_row;
        int64_t const column = value_offset + index - row * k;
        float const gate = static_cast<float>(input[row * 2 * k + column]);
        float const up = static_cast<float>(input[row * 2 * k + k + column]);
        float const activated = static_cast<float>(static_cast<Source>(
            gate / (1.0f + expf(-gate))));
        // The separate activation's up + beta (beta=+0) canonicalizes -0.
        // Make that boundary explicit rather than depending on fast-math.
        float const shifted_up = up == 0.0f ? 0.0f : up;
        values[index] = static_cast<float>(static_cast<Source>(activated * shifted_up));
      } else {
        values[index] = static_cast<float>(input[value_offset + index]);
      }
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
    if constexpr (PackedScaleLayout) {
      int const row = static_cast<int>(group / groups_per_row);
      int const k_group =
          static_cast<int>(group -
                           static_cast<int64_t>(row) * groups_per_row);
      auto const scale_offset = scale_layout(cute::make_coord(
          row, k_group * kScaleVectorSize, 0));
      scales[scale_offset] = scale_code;
      if (logical_scales != nullptr) {
        logical_scales[group] = scale_code;
      }
    } else {
      scales[group] = scale_code;
    }
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

template <int OutputBits, bool PackedScaleLayout = true, bool SiluAndMul = false>
std::tuple<at::Tensor, at::Tensor> quantize_mx(
    at::Tensor const& input) {
  check_quant_input(input);
  c10::cuda::CUDAGuard guard(input.device());
  int64_t const m = input.size(0);
  int64_t const k = input.size(1) / (SiluAndMul ? 2 : 1);
  TORCH_CHECK(k > 0 && k % kScaleVectorSize == 0, "output K must be divisible by 32");
  int64_t const values = m * k;
  int64_t const groups_per_row = k / kScaleVectorSize;
  int64_t const packed_groups_per_row = round_up(groups_per_row, 4);
  int64_t const total_groups = m * groups_per_row;
  int64_t const padded_rows = round_up(m, 128);

  auto byte_options = input.options().dtype(at::kByte);
  at::Tensor output = OutputBits == 8
      ? at::empty({m, k}, byte_options)
      : at::empty({values * 3 / 4}, byte_options);
  at::Tensor scales = PackedScaleLayout
      ? at::empty({padded_rows * packed_groups_per_row}, byte_options)
      : at::empty({m, groups_per_row}, byte_options);

  auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());

  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;
  auto const scale_layout = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(static_cast<int>(m), 1, static_cast<int>(k), 1));
  int64_t const block_count = PackedScaleLayout
      ? std::max(ceil_div(total_groups, kGroupsPerBlock),
                 ceil_div(padded_rows * packed_groups_per_row, kThreads))
      : ceil_div(total_groups, kGroupsPerBlock);
  TORCH_CHECK(block_count <= std::numeric_limits<int>::max(),
              "quantization launch grid is too large");

  if (input.scalar_type() == at::kHalf) {
    quantize_mx_kernel<at::Half, OutputBits, PackedScaleLayout, decltype(scale_layout), SiluAndMul><<<
        static_cast<int>(block_count), kThreads, 0, stream.stream()>>>(
        input.data_ptr<at::Half>(), output.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(), nullptr,
        static_cast<int>(groups_per_row), total_groups, scale_layout, PackedScaleLayout);
  } else {
    quantize_mx_kernel<at::BFloat16, OutputBits, PackedScaleLayout, decltype(scale_layout), SiluAndMul><<<
        static_cast<int>(block_count), kThreads, 0, stream.stream()>>>(
        input.data_ptr<at::BFloat16>(), output.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(), nullptr,
        static_cast<int>(groups_per_row), total_groups, scale_layout, PackedScaleLayout);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> quantize_mxfp8_cuda(
    at::Tensor const& input) {
  return quantize_mx<8>(input);
}

std::tuple<at::Tensor, at::Tensor> quantize_mxfp8_logical_cuda(
    at::Tensor const& input) {
  return quantize_mx<8, false>(input);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_mxfp8_dual_cuda(at::Tensor const& input) {
  check_quant_input(input);
  c10::cuda::CUDAGuard guard(input.device());
  int64_t const m = input.size(0);
  int64_t const k = input.size(1);
  int64_t const groups_per_row = k / kScaleVectorSize;
  int64_t const packed_groups_per_row = round_up(groups_per_row, 4);
  int64_t const total_groups = m * groups_per_row;
  int64_t const padded_rows = round_up(m, 128);
  auto byte_options = input.options().dtype(at::kByte);
  auto output = at::empty({m, k}, byte_options);
  auto logical_scales =
      at::empty({m, groups_per_row}, byte_options);
  auto packed_scales = at::empty(
      {padded_rows * packed_groups_per_row}, byte_options);

  auto stream =
      c10::cuda::getCurrentCUDAStream(input.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      packed_scales.data_ptr<uint8_t>(), kUe8m0One,
      static_cast<size_t>(packed_scales.numel()), stream.stream()));
  using ScaleConfig =
      cutlass::detail::Sm1xxBlockScaledConfig<32>;
  auto const scale_layout =
      ScaleConfig::tile_atom_to_shape_SFA(
          cute::make_shape(
              static_cast<int>(m), 1,
              static_cast<int>(k), 1));
  int64_t const block_count =
      ceil_div(total_groups, kGroupsPerBlock);
  TORCH_CHECK(
      block_count <= std::numeric_limits<int>::max(),
      "quantization launch grid is too large");

  if (input.scalar_type() == at::kHalf) {
    quantize_mx_kernel<at::Half, 8, true><<<
        static_cast<int>(block_count),
        kThreads,
        0,
        stream.stream()>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<uint8_t>(),
            packed_scales.data_ptr<uint8_t>(),
            logical_scales.data_ptr<uint8_t>(),
            static_cast<int>(groups_per_row),
            total_groups,
            scale_layout);
  } else {
    quantize_mx_kernel<at::BFloat16, 8, true><<<
        static_cast<int>(block_count),
        kThreads,
        0,
        stream.stream()>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<uint8_t>(),
            packed_scales.data_ptr<uint8_t>(),
            logical_scales.data_ptr<uint8_t>(),
            static_cast<int>(groups_per_row),
            total_groups,
            scale_layout);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, logical_scales, packed_scales};
}

void quantize_mxfp8_dual_out_cuda(
    at::Tensor& output,
    at::Tensor& logical_scales,
    at::Tensor& packed_scales,
    at::Tensor const& input) {
  check_quant_input(input);
  c10::cuda::CUDAGuard guard(input.device());
  auto const device = input.device();
  int64_t const m = input.size(0);
  int64_t const k = input.size(1);
  int64_t const groups_per_row = k / kScaleVectorSize;
  int64_t const packed_groups_per_row =
      round_up(groups_per_row, 4);
  int64_t const total_groups = m * groups_per_row;
  int64_t const padded_rows = round_up(m, 128);
  TORCH_CHECK(
      output.is_cuda() && output.device() == device &&
          output.scalar_type() == at::kByte &&
          output.is_contiguous() &&
          output.sizes() == input.sizes(),
      "output must be contiguous CUDA uint8 [M,K]");
  TORCH_CHECK(
      logical_scales.is_cuda() &&
          logical_scales.device() == device &&
          logical_scales.scalar_type() == at::kByte &&
          logical_scales.is_contiguous() &&
          logical_scales.dim() == 2 &&
          logical_scales.size(0) == m &&
          logical_scales.size(1) == groups_per_row,
      "logical_scales must be contiguous CUDA uint8 [M,K/32]");
  TORCH_CHECK(
      packed_scales.is_cuda() &&
          packed_scales.device() == device &&
          packed_scales.scalar_type() == at::kByte &&
          packed_scales.is_contiguous() &&
          packed_scales.numel() >=
              padded_rows * packed_groups_per_row,
      "packed_scales is too small for the SM120 scale layout");

  using ScaleConfig =
      cutlass::detail::Sm1xxBlockScaledConfig<32>;
  auto const scale_layout =
      ScaleConfig::tile_atom_to_shape_SFA(
          cute::make_shape(
              static_cast<int>(m), 1,
              static_cast<int>(k), 1));
  int64_t const block_count =
      ceil_div(total_groups, kGroupsPerBlock);
  TORCH_CHECK(
      block_count <= std::numeric_limits<int>::max(),
      "quantization launch grid is too large");
  auto stream =
      c10::cuda::getCurrentCUDAStream(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    quantize_mx_kernel<at::Half, 8, true><<<
        static_cast<int>(block_count),
        kThreads,
        0,
        stream.stream()>>>(
            input.data_ptr<at::Half>(),
            output.data_ptr<uint8_t>(),
            packed_scales.data_ptr<uint8_t>(),
            logical_scales.data_ptr<uint8_t>(),
            static_cast<int>(groups_per_row),
            total_groups,
            scale_layout);
  } else {
    quantize_mx_kernel<at::BFloat16, 8, true><<<
        static_cast<int>(block_count),
        kThreads,
        0,
        stream.stream()>>>(
            input.data_ptr<at::BFloat16>(),
            output.data_ptr<uint8_t>(),
            packed_scales.data_ptr<uint8_t>(),
            logical_scales.data_ptr<uint8_t>(),
            static_cast<int>(groups_per_row),
            total_groups,
            scale_layout);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor> quantize_mxfp6_cuda(
    at::Tensor const& input) {
  return quantize_mx<6>(input);
}

}  // namespace mxfp6_gemm::torch_ext

namespace mxfp6_gemm::torch_ext {
std::tuple<at::Tensor, at::Tensor> silu_and_mul_mxfp8_cuda(at::Tensor const& input) {
  TORCH_CHECK(input.dim() == 2 && input.size(1) % 64 == 0,
              "SwiGLU input must be [M,2K] with K divisible by 32");
  return quantize_mx<8, true, true>(input);
}
}  // namespace mxfp6_gemm::torch_ext

TORCH_LIBRARY_FRAGMENT(mxfp6, m) {
  m.def("silu_and_mul_mxfp8(Tensor input) -> (Tensor values, Tensor scales)");
}
TORCH_LIBRARY_IMPL(mxfp6, CUDA, m) {
  m.impl("silu_and_mul_mxfp8", &mxfp6_gemm::torch_ext::silu_and_mul_mxfp8_cuda);
}
