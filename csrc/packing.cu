#include "mxfp6_gemm/packing.hpp"

#include <cstdint>
#include <limits>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

namespace mxfp6_gemm::torch_ext {
namespace {

constexpr int kThreads = 256;
constexpr int kScaleVectorSize = 32;
constexpr uint8_t kUe8m0One = 0x7f;

int64_t ceil_div(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

int64_t round_up(int64_t value, int64_t alignment) {
  return ceil_div(value, alignment) * alignment;
}

void check_cuda_byte_tensor(at::Tensor const& tensor, char const* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kByte,
              name, " must have dtype torch.uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_matrix_shape(int64_t rows, int64_t k) {
  TORCH_CHECK(rows > 0, "rows must be positive; got ", rows);
  TORCH_CHECK(k > 0, "k must be positive; got ", k);
  TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                  k <= std::numeric_limits<int>::max(),
              "rows and k must fit in a 32-bit integer");
}

__device__ __forceinline__ uint32_t pack_four_codes(uint32_t codes) {
  uint32_t const c0 = codes & 0x3fu;
  uint32_t const c1 = (codes >> 8) & 0x3fu;
  uint32_t const c2 = (codes >> 16) & 0x3fu;
  uint32_t const c3 = (codes >> 24) & 0x3fu;
  return c0 | (c1 << 6) | (c2 << 12) | (c3 << 18);
}

__device__ __forceinline__ uint32_t unpack_four_codes(uint32_t packed) {
  return (packed & 0x3fu) |
      (((packed >> 6) & 0x3fu) << 8) |
      (((packed >> 12) & 0x3fu) << 16) |
      (((packed >> 18) & 0x3fu) << 24);
}

// One thread converts 16 input bytes to 12 output bytes. The input uses one
// aligned 128-bit transaction and the output uses three aligned 32-bit stores.
__global__ void pack_fp6_vector_kernel(
    uint8_t const* input, uint8_t* output, int64_t vectors) {
  int64_t const vector =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vector >= vectors) {
    return;
  }

  uint4 const raw = reinterpret_cast<uint4 const*>(input)[vector];
  uint32_t const p0 = pack_four_codes(raw.x);
  uint32_t const p1 = pack_four_codes(raw.y);
  uint32_t const p2 = pack_four_codes(raw.z);
  uint32_t const p3 = pack_four_codes(raw.w);

  uint32_t* destination =
      reinterpret_cast<uint32_t*>(output + vector * 12);
  destination[0] = p0 | ((p1 & 0xffu) << 24);
  destination[1] = (p1 >> 8) | ((p2 & 0xffffu) << 16);
  destination[2] = (p2 >> 16) | (p3 << 8);
}

__global__ void pack_fp6_scalar_kernel(
    uint8_t const* input,
    uint8_t* output,
    int64_t first_group,
    int64_t groups) {
  int64_t const local_group =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_group >= groups) {
    return;
  }
  int64_t const group = first_group + local_group;
  uint8_t const* source = input + group * 4;
  uint32_t const packed =
      (source[0] & 0x3f) |
      ((source[1] & 0x3f) << 6) |
      ((source[2] & 0x3f) << 12) |
      ((source[3] & 0x3f) << 18);
  uint8_t* destination = output + group * 3;
  destination[0] = packed;
  destination[1] = packed >> 8;
  destination[2] = packed >> 16;
}

__global__ void unpack_fp6_vector_kernel(
    uint8_t const* input, uint8_t* output, int64_t vectors) {
  int64_t const vector =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vector >= vectors) {
    return;
  }

  uint32_t const* source =
      reinterpret_cast<uint32_t const*>(input + vector * 12);
  uint32_t const i0 = source[0];
  uint32_t const i1 = source[1];
  uint32_t const i2 = source[2];
  uint32_t const p0 = i0 & 0x00ffffffu;
  uint32_t const p1 = (i0 >> 24) | ((i1 & 0x0000ffffu) << 8);
  uint32_t const p2 = (i1 >> 16) | ((i2 & 0x000000ffu) << 16);
  uint32_t const p3 = i2 >> 8;

  uint4 result{
      unpack_four_codes(p0),
      unpack_four_codes(p1),
      unpack_four_codes(p2),
      unpack_four_codes(p3)};
  reinterpret_cast<uint4*>(output)[vector] = result;
}

__global__ void unpack_fp6_scalar_kernel(
    uint8_t const* input,
    uint8_t* output,
    int64_t first_group,
    int64_t groups) {
  int64_t const local_group =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local_group >= groups) {
    return;
  }
  int64_t const group = first_group + local_group;
  uint8_t const* source = input + group * 3;
  uint32_t const packed = static_cast<uint32_t>(source[0]) |
      (static_cast<uint32_t>(source[1]) << 8) |
      (static_cast<uint32_t>(source[2]) << 16);
  uint8_t* destination = output + group * 4;
  destination[0] = packed & 0x3f;
  destination[1] = (packed >> 6) & 0x3f;
  destination[2] = (packed >> 12) & 0x3f;
  destination[3] = (packed >> 18) & 0x3f;
}

template <class Layout>
__global__ void pack_scales_kernel(
    uint8_t const* logical,
    uint8_t* packed,
    int rows,
    int padded_rows,
    int k_blocks,
    Layout layout) {
  int64_t const linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const elements =
      static_cast<int64_t>(padded_rows) * k_blocks;
  if (linear >= elements) {
    return;
  }
  int const row = linear / k_blocks;
  int const k_block = linear - row * k_blocks;
  // The layout's K coordinate is in values, not scale blocks. The atom has a
  // zero stride across the 32 values sharing one UE8M0 byte.
  auto const offset = layout(cute::make_coord(
      row, k_block * kScaleVectorSize, 0));
  packed[offset] = row < rows
      ? logical[static_cast<int64_t>(row) * k_blocks + k_block]
      : kUe8m0One;
}

template <class Layout>
__global__ void unpack_scales_kernel(
    uint8_t const* packed,
    uint8_t* logical,
    int rows,
    int k_blocks,
    Layout layout) {
  int64_t const linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const elements = static_cast<int64_t>(rows) * k_blocks;
  if (linear >= elements) {
    return;
  }
  int const row = linear / k_blocks;
  int const k_block = linear - row * k_blocks;
  auto const offset = layout(cute::make_coord(
      row, k_block * kScaleVectorSize, 0));
  logical[linear] = packed[offset];
}

template <class Kernel, class... Args>
void launch_1d(int64_t count, cudaStream_t stream, Kernel kernel, Args... args) {
  if (count == 0) {
    return;
  }
  int64_t const block_count = ceil_div(count, kThreads);
  TORCH_CHECK(block_count <= std::numeric_limits<int>::max(),
              "CUDA launch grid is too large");
  kernel<<<static_cast<int>(block_count), kThreads, 0, stream>>>(args...);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor pack_fp6_cuda(at::Tensor const& codes) {
  check_cuda_byte_tensor(codes, "codes");
  TORCH_CHECK(codes.numel() % 4 == 0,
              "codes.numel() must be divisible by four; got ", codes.numel());
  c10::cuda::CUDAGuard guard(codes.device());
  auto output = at::empty({codes.numel() * 3 / 4}, codes.options());
  auto stream = c10::cuda::getCurrentCUDAStream(codes.get_device());

  int64_t const values = codes.numel();
  bool const vector_aligned =
      reinterpret_cast<uintptr_t>(codes.data_ptr()) % alignof(uint4) == 0;
  int64_t const vectors = vector_aligned ? values / 16 : 0;
  launch_1d(vectors, stream.stream(), pack_fp6_vector_kernel,
            codes.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(), vectors);
  int64_t const first_group = vectors * 4;
  int64_t const tail_groups = values / 4 - first_group;
  launch_1d(tail_groups, stream.stream(), pack_fp6_scalar_kernel,
            codes.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(),
            first_group, tail_groups);
  return output;
}

at::Tensor unpack_fp6_cuda(
    at::Tensor const& packed, int64_t rows, int64_t k) {
  check_cuda_byte_tensor(packed, "packed");
  check_matrix_shape(rows, k);
  TORCH_CHECK(k % 4 == 0, "k must be divisible by four; got ", k);
  TORCH_CHECK(rows <= std::numeric_limits<int64_t>::max() / k,
              "rows * k overflows int64");
  int64_t const values = rows * k;
  TORCH_CHECK(packed.numel() == values * 3 / 4,
              "packed must contain exactly ", values * 3 / 4,
              " bytes; got ", packed.numel());

  c10::cuda::CUDAGuard guard(packed.device());
  auto output = at::empty({rows, k}, packed.options());
  auto stream = c10::cuda::getCurrentCUDAStream(packed.get_device());
  bool const vector_aligned =
      reinterpret_cast<uintptr_t>(packed.data_ptr()) % alignof(uint32_t) == 0;
  int64_t const vectors = vector_aligned ? values / 16 : 0;
  launch_1d(vectors, stream.stream(), unpack_fp6_vector_kernel,
            packed.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(), vectors);
  int64_t const first_group = vectors * 4;
  int64_t const tail_groups = values / 4 - first_group;
  launch_1d(tail_groups, stream.stream(), unpack_fp6_scalar_kernel,
            packed.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(),
            first_group, tail_groups);
  return output;
}

at::Tensor pack_scales_cuda(
    at::Tensor const& logical, int64_t rows, int64_t k) {
  check_cuda_byte_tensor(logical, "logical scales");
  check_matrix_shape(rows, k);
  TORCH_CHECK(k % 128 == 0,
              "k must be divisible by 128 for the SM120 scale atom; got ", k);
  int64_t const k_blocks = k / kScaleVectorSize;
  TORCH_CHECK(logical.numel() == rows * k_blocks,
              "logical scales must contain exactly ", rows * k_blocks,
              " bytes; got ", logical.numel());

  int64_t const padded_rows = round_up(rows, 128);
  auto output = at::empty({padded_rows * k_blocks}, logical.options());
  c10::cuda::CUDAGuard guard(logical.device());
  auto stream = c10::cuda::getCurrentCUDAStream(logical.get_device());
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;
  auto const layout = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(
      static_cast<int>(rows), 1, static_cast<int>(k), 1));
  int64_t const elements = padded_rows * k_blocks;
  launch_1d(elements, stream.stream(), pack_scales_kernel<decltype(layout)>,
            logical.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(),
            static_cast<int>(rows), static_cast<int>(padded_rows),
            static_cast<int>(k_blocks), layout);
  return output;
}

at::Tensor unpack_scales_cuda(
    at::Tensor const& packed, int64_t rows, int64_t k) {
  check_cuda_byte_tensor(packed, "packed scales");
  check_matrix_shape(rows, k);
  TORCH_CHECK(k % 128 == 0,
              "k must be divisible by 128 for the SM120 scale atom; got ", k);
  int64_t const k_blocks = k / kScaleVectorSize;
  int64_t const packed_size = round_up(rows, 128) * k_blocks;
  TORCH_CHECK(packed.numel() == packed_size,
              "packed scales must contain exactly ", packed_size,
              " bytes; got ", packed.numel());

  c10::cuda::CUDAGuard guard(packed.device());
  auto output = at::empty({rows, k_blocks}, packed.options());
  auto stream = c10::cuda::getCurrentCUDAStream(packed.get_device());
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;
  auto const layout = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(
      static_cast<int>(rows), 1, static_cast<int>(k), 1));
  int64_t const elements = rows * k_blocks;
  launch_1d(elements, stream.stream(), unpack_scales_kernel<decltype(layout)>,
            packed.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(),
            static_cast<int>(rows), static_cast<int>(k_blocks), layout);
  return output;
}

}  // namespace mxfp6_gemm::torch_ext
