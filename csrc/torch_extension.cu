#include <cstdint>
#include <limits>
#include <string>

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "mxfp6_gemm/kernel_normal.hpp"
#include "mxfp6_gemm/kernel_swapped.hpp"
#include "mxfp6_gemm/packing.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

enum class RasterOrder {
  Heuristic,
  AlongM,
  AlongN,
};

template <class Kernel, class Arguments>
void configure_scheduler(Arguments& arguments,
                         int max_swizzle_size,
                         RasterOrder raster_order,
                         int splits) {
  using SchedulerRaster = decltype(arguments.scheduler.raster_order);
  arguments.scheduler.max_swizzle_size = max_swizzle_size;
  if (raster_order == RasterOrder::AlongN) {
    arguments.scheduler.raster_order = SchedulerRaster::AlongN;
  } else if (raster_order == RasterOrder::AlongM) {
    arguments.scheduler.raster_order = SchedulerRaster::AlongM;
  } else {
    arguments.scheduler.raster_order = SchedulerRaster::Heuristic;
  }

  if constexpr (Kernel::IsStreamK) {
    using DecompositionMode = cutlass::gemm::kernel::detail::
        PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    arguments.scheduler.splits = splits;
    arguments.scheduler.decomposition_mode = splits > 1
        ? DecompositionMode::SplitK
        : DecompositionMode::Heuristic;
  }
}

int64_t round_up(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

void check_input(at::Tensor const& tensor,
                 char const* name,
                 c10::Device device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device,
              name, " must be on ", device, "; got ", tensor.device());
  TORCH_CHECK(tensor.scalar_type() == at::kByte,
              name, " must have dtype torch.uint8");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
              name, " must be at least 16-byte aligned");
}

template <class Kernel>
at::Tensor launch_swapped(at::Tensor const& a,
                          at::Tensor const& b,
                          at::Tensor const& sfa,
                          at::Tensor const& sfb,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          double alpha,
                          int device_index,
                          int max_swizzle_size = 8,
                          RasterOrder raster_order = RasterOrder::Heuristic,
                          int splits = 1) {
  using Gemm = typename Kernel::Gemm;
  using BlockScaledConfig = typename Kernel::BlockScaledConfig;
  auto output = at::empty({m, n}, a.options().dtype(at::kHalf));
  auto problem = cute::make_shape(
      static_cast<int>(n), static_cast<int>(m), static_cast<int>(k), 1);
  auto stride_a = cutlass::make_cute_packed_stride(
      typename Kernel::StrideA{}, cute::make_shape(
          static_cast<int>(n), static_cast<int>(k), 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      typename Kernel::StrideB{}, cute::make_shape(
          static_cast<int>(m), static_cast<int>(k), 1));
  auto stride_c = cutlass::make_cute_packed_stride(
      typename Kernel::StrideC{}, cute::make_shape(
          static_cast<int>(n), static_cast<int>(m), 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      typename Kernel::StrideD{}, cute::make_shape(
          static_cast<int>(n), static_cast<int>(m), 1));
  auto layout_sfa = BlockScaledConfig::tile_atom_to_shape_SFA(problem);
  auto layout_sfb = BlockScaledConfig::tile_atom_to_shape_SFB(problem);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {reinterpret_cast<typename Kernel::ElementA const*>(b.data_ptr<uint8_t>()),
       stride_a,
       reinterpret_cast<typename Kernel::ElementB const*>(a.data_ptr<uint8_t>()),
       stride_b,
       reinterpret_cast<typename Kernel::ElementSF const*>(sfb.data_ptr<uint8_t>()),
       layout_sfa,
       reinterpret_cast<typename Kernel::ElementSF const*>(sfa.data_ptr<uint8_t>()),
       layout_sfb},
      {{static_cast<float>(alpha), 0.0f},
       static_cast<typename Kernel::ElementC const*>(nullptr), stride_c,
       reinterpret_cast<typename Kernel::ElementD*>(output.data_ptr<at::Half>()),
       stride_d}};

  configure_scheduler<Kernel>(
      arguments, max_swizzle_size, raster_order, splits);

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS can_implement failed: ", cutlassGetStatusString(status));

  size_t const workspace_bytes = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_bytes)}, a.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }

  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  status = gemm.initialize(arguments, workspace_ptr, stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS initialize failed: ", cutlassGetStatusString(status));
  status = gemm.run(stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS launch failed: ", cutlassGetStatusString(status));
  if (workspace.defined()) {
    c10::cuda::CUDACachingAllocator::recordStream(
        workspace.storage().data_ptr(), stream);
  }
  return output;
}

template <class Kernel>
at::Tensor launch_normal(at::Tensor const& a,
                         at::Tensor const& b,
                         at::Tensor const& sfa,
                         at::Tensor const& sfb,
                         int64_t m,
                         int64_t n,
                         int64_t k,
                         double alpha,
                         int device_index,
                         int max_swizzle_size = 8,
                         RasterOrder raster_order = RasterOrder::Heuristic,
                         int splits = 1) {
  using Gemm = typename Kernel::Gemm;
  using BlockScaledConfig = typename Kernel::BlockScaledConfig;
  auto output = at::empty({m, n}, a.options().dtype(at::kHalf));
  auto problem = cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1);
  auto stride_a = cutlass::make_cute_packed_stride(
      typename Kernel::StrideA{}, cute::make_shape(
          static_cast<int>(m), static_cast<int>(k), 1));
  auto stride_b = cutlass::make_cute_packed_stride(
      typename Kernel::StrideB{}, cute::make_shape(
          static_cast<int>(n), static_cast<int>(k), 1));
  auto stride_c = cutlass::make_cute_packed_stride(
      typename Kernel::StrideC{}, cute::make_shape(
          static_cast<int>(m), static_cast<int>(n), 1));
  auto stride_d = cutlass::make_cute_packed_stride(
      typename Kernel::StrideD{}, cute::make_shape(
          static_cast<int>(m), static_cast<int>(n), 1));
  auto layout_sfa = BlockScaledConfig::tile_atom_to_shape_SFA(problem);
  auto layout_sfb = BlockScaledConfig::tile_atom_to_shape_SFB(problem);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {reinterpret_cast<typename Kernel::ElementA const*>(a.data_ptr<uint8_t>()),
       stride_a,
       reinterpret_cast<typename Kernel::ElementB const*>(b.data_ptr<uint8_t>()),
       stride_b,
       reinterpret_cast<typename Kernel::ElementSF const*>(sfa.data_ptr<uint8_t>()),
       layout_sfa,
       reinterpret_cast<typename Kernel::ElementSF const*>(sfb.data_ptr<uint8_t>()),
       layout_sfb},
      {{static_cast<float>(alpha), 0.0f},
       static_cast<typename Kernel::ElementC const*>(nullptr), stride_c,
       reinterpret_cast<typename Kernel::ElementD*>(output.data_ptr<at::Half>()),
       stride_d}};

  configure_scheduler<Kernel>(
      arguments, max_swizzle_size, raster_order, splits);

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS can_implement failed: ", cutlassGetStatusString(status));

  size_t const workspace_bytes = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_bytes)}, a.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }

  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  status = gemm.initialize(arguments, workspace_ptr, stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS initialize failed: ", cutlassGetStatusString(status));
  status = gemm.run(stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS launch failed: ", cutlassGetStatusString(status));
  if (workspace.defined()) {
    c10::cuda::CUDACachingAllocator::recordStream(
        workspace.storage().data_ptr(), stream);
  }
  return output;
}

at::Tensor launch_swapped_policy(at::Tensor const& a,
                                 at::Tensor const& b,
                                 at::Tensor const& sfa,
                                 at::Tensor const& sfb,
                                 int64_t m,
                                 int64_t n,
                                 int64_t k,
                                 double alpha,
                                 int device_index) {
  namespace kernels = mxfp6_gemm::swapped;
  int64_t const output_elements = m * n;
  int const stream_k_splits = output_elements <= 524288 ? 4 : 1;

  if (k >= 8192) {
    if (m <= 8) {
      return launch_swapped<kernels::Kernel128x8StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::Heuristic, stream_k_splits);
    }
    if (m <= 16) {
      return launch_swapped<kernels::Kernel128x16StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::Heuristic, stream_k_splits);
    }
    if (output_elements <= 65536) {
      if (m <= 64 && k <= 8192) {
        return launch_swapped<kernels::Kernel64x16x256Stage3Pingpong>(
            a, b, sfa, sfb, m, n, k, alpha, device_index);
      }
      return launch_swapped<kernels::Kernel128x16StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::Heuristic, stream_k_splits);
    }
    if (output_elements <= 524288 || (m > 64 && n >= 8192)) {
      return launch_swapped<kernels::Kernel128x32Stage3StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::Heuristic, stream_k_splits);
    }
    return launch_swapped<kernels::Kernel128x64StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        8, RasterOrder::Heuristic, stream_k_splits);
  }

  if (k >= 2048) {
    if (n <= 512) {
      return launch_swapped<kernels::Kernel64x16x256Stage3Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    if (m <= 8 && n >= 8192) {
      return launch_swapped<kernels::Kernel128x8Stage4Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    if (m <= 16) {
      if (n <= 8192 && k <= 2048) {
        return launch_swapped<kernels::Kernel64x16x256Stage3Pingpong>(
            a, b, sfa, sfb, m, n, k, alpha, device_index);
      }
      if (n >= 8192) {
        return launch_swapped<kernels::Kernel128x16StreamK>(
            a, b, sfa, sfb, m, n, k, alpha, device_index,
            8, RasterOrder::Heuristic, stream_k_splits);
      }
      return launch_swapped<kernels::Kernel64x16x128Stage3Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    if (n >= 8192) {
      if (m <= 32) {
        if (n <= 8192 && k <= 2048) {
          return launch_swapped<kernels::Kernel64x32x128Stage3Pingpong>(
              a, b, sfa, sfb, m, n, k, alpha, device_index);
        }
        return launch_swapped<kernels::Kernel128x32Stage3StreamK>(
            a, b, sfa, sfb, m, n, k, alpha, device_index,
            8, RasterOrder::Heuristic, stream_k_splits);
      }
      if (m > 64 || output_elements <= 524288) {
        return launch_swapped<kernels::Kernel128x32Stage3StreamK>(
            a, b, sfa, sfb, m, n, k, alpha, device_index,
            8, RasterOrder::Heuristic, stream_k_splits);
      }
      return launch_swapped<kernels::Kernel128x64StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::Heuristic, stream_k_splits);
    }
    if (m > 64 && k >= 4096) {
      return launch_swapped<kernels::Kernel128x32Stage3StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::Heuristic, stream_k_splits);
    }
    return launch_swapped<kernels::Kernel64x32x128Stage3Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index);
  }

  if (m <= 8 && n >= 8192) {
    return launch_swapped<kernels::Kernel128x8Stage2Cooperative>(
        a, b, sfa, sfb, m, n, k, alpha, device_index);
  }
  if (m <= 16 && n >= 8192) {
    return launch_swapped<kernels::Kernel64x16x128Stage3Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index);
  }
  if (m <= 64 && output_elements > 524288) {
    return launch_swapped<kernels::Kernel128x64StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        8, RasterOrder::Heuristic, 1);
  }
  return launch_swapped<kernels::Kernel64x32x128Stage3Pingpong>(
      a, b, sfa, sfb, m, n, k, alpha, device_index);
}

at::Tensor launch_normal_policy(at::Tensor const& a,
                                at::Tensor const& b,
                                at::Tensor const& sfa,
                                at::Tensor const& sfb,
                                int64_t m,
                                int64_t n,
                                int64_t k,
                                double alpha,
                                int device_index) {
  namespace kernels = mxfp6_gemm::normal;
  int64_t const output_elements = m * n;
  int const stream_k_splits = output_elements <= 524288 ? 4 : 1;

  if (k <= 512) {
    if (output_elements <= 262144) {
      return launch_normal<kernels::Kernel64x32x128Stage3Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    if (output_elements <= 16777216) {
      return launch_normal<kernels::Kernel64x64x128Stage2Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    return launch_normal<kernels::Kernel128x128x128Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index);
  }

  if (k < 8192) {
    if (output_elements <= 131072 && n <= 1024) {
      return launch_normal<kernels::Kernel64x16x256Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    if (output_elements <= 262144) {
      return launch_normal<kernels::Kernel64x32x128Stage3Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    if (output_elements <= 16777216) {
      return launch_normal<kernels::Kernel64x64x128Stage4Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    return launch_normal<kernels::Kernel128x128x128Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index);
  }

  if (n <= 128) {
    if (m <= 512) {
      if (k >= 16384) {
        return launch_normal<kernels::Kernel128x16x128Stage3StreamK>(
            a, b, sfa, sfb, m, n, k, alpha, device_index,
            8, RasterOrder::Heuristic, stream_k_splits);
      }
      return launch_normal<kernels::Kernel64x16x256Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    if (m <= 2048) {
      return launch_normal<kernels::Kernel128x32x128StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::Heuristic, stream_k_splits);
    }
    if (m <= 4096) {
      return launch_normal<kernels::Kernel128x64x128StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::Heuristic, stream_k_splits);
    }
    return launch_normal<kernels::Kernel128x128x128StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        8, RasterOrder::Heuristic, stream_k_splits);
  }
  if (output_elements <= 65536) {
    return launch_normal<kernels::Kernel64x16x256Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index);
  }
  if (output_elements <= 262144) {
    return launch_normal<kernels::Kernel128x32x128StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        8, RasterOrder::Heuristic, stream_k_splits);
  }
  if (output_elements <= 4194304) {
    return launch_normal<kernels::Kernel128x64x128StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        8, RasterOrder::Heuristic, stream_k_splits);
  }
  return launch_normal<kernels::Kernel128x128x128StreamK>(
      a, b, sfa, sfb, m, n, k, alpha, device_index,
      8, RasterOrder::Heuristic, stream_k_splits);
}

at::Tensor gemm_cuda_impl(at::Tensor const& a,
                          at::Tensor const& b,
                          at::Tensor const& sfa,
                          at::Tensor const& sfb,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          double alpha) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  TORCH_CHECK(false, "mxfp6_torch must be compiled for sm_120a");
#else
  TORCH_CHECK(m > 0, "m must be positive; got ", m);
  TORCH_CHECK(n > 0 && n % 8 == 0,
              "n must be a positive multiple of 8; got ", n);
  TORCH_CHECK(k > 0 && k % 128 == 0,
              "k must be a positive multiple of 128; got ", k);
  TORCH_CHECK(m <= std::numeric_limits<int>::max() &&
                  n <= std::numeric_limits<int>::max() &&
                  k <= std::numeric_limits<int>::max(),
              "m, n, and k must fit in a 32-bit integer");
  TORCH_CHECK(m <= std::numeric_limits<int64_t>::max() / k &&
                  n <= std::numeric_limits<int64_t>::max() / k,
              "packed operand size exceeds int64 range");

  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(a.device());
  check_input(a, "a", a.device());
  check_input(b, "b", a.device());
  check_input(sfa, "sfa", a.device());
  check_input(sfb, "sfb", a.device());

  // Four FP6 values occupy three bytes. Both buffers are packed from logical
  // [M,K] / [N,K] order; CUTLASS's B stride interprets the latter as the
  // column-major operand of A @ B.T.
  int64_t const a_bytes = (m * k / 4) * 3;
  int64_t const b_bytes = (n * k / 4) * 3;
  int64_t const sfa_bytes = round_up(m, 128) * k / 32;
  int64_t const sfb_bytes = round_up(n, 128) * k / 32;
  TORCH_CHECK(a.numel() == a_bytes,
              "a must contain exactly ", a_bytes, " packed bytes; got ",
              a.numel());
  TORCH_CHECK(b.numel() == b_bytes,
              "b must contain exactly ", b_bytes, " packed bytes; got ",
              b.numel());
  TORCH_CHECK(sfa.numel() == sfa_bytes,
              "sfa must contain exactly ", sfa_bytes,
              " bytes in CUTLASS block-scale layout; got ", sfa.numel());
  TORCH_CHECK(sfb.numel() == sfb_bytes,
              "sfb must contain exactly ", sfb_bytes,
              " bytes in CUTLASS block-scale layout; got ", sfb.numel());

  cudaDeviceProp properties{};
  int device_index = a.get_device();
  auto cuda_status = cudaGetDeviceProperties(&properties, device_index);
  TORCH_CHECK(cuda_status == cudaSuccess,
              "cudaGetDeviceProperties failed: ",
              cudaGetErrorString(cuda_status));
  TORCH_CHECK(properties.major == 12 && properties.minor == 0,
              "mxfp6::gemm requires SM120; current device is SM",
              properties.major, properties.minor);

  bool const target_nk =
      (n == 5120 && k == 8192) ||
      (n == 3072 && k == 5120) ||
      (n == 5120 && k == 7168) ||
      (n == 5120 && k == 17408) ||
      (n == 8704 && k == 5120);

  // Retain the exact cold-cache winners for the original benchmark set.
  if (m == 2048 && target_nk) {
    if (n == 3072 && k == 5120) {
      return launch_normal<
          mxfp6_gemm::normal::TargetKernel64x128x128Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongN);
    }
    if (n == 5120 && (k == 7168 || k == 17408)) {
      return launch_normal<
          mxfp6_gemm::normal::TargetKernel128x128x128Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongN);
    }
    if (n == 5120 && k == 8192) {
      return launch_normal<mxfp6_gemm::normal::Kernel128x128x128Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          2, RasterOrder::AlongM);
    }
    return launch_normal<mxfp6_gemm::normal::Kernel128x128x128StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, RasterOrder::AlongM, 1);
  }

  if (target_nk && m == 1) {
    return launch_swapped<mxfp6_gemm::swapped::TargetKernel128x8StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, RasterOrder::AlongM, 2);
  }
  if (target_nk && m == 16) {
    if (n == 8704 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM);
    }
    return launch_swapped<mxfp6_gemm::swapped::TargetKernel128x8StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, RasterOrder::AlongM, 2);
  }
  if (target_nk && m == 32) {
    if (n == 8704 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x16Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM);
    }
    return launch_swapped<mxfp6_gemm::swapped::TargetKernel128x16StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, RasterOrder::AlongM, 2);
  }

  // Swapping maps the small M dimension to the tensor-core N tile. Boundary
  // profiling favors this orientation through M=96 and normal order from the
  // next region. The policy helpers then select among the compact profiler-
  // derived portfolio using K depth and output-tile count.
  if (m <= 96) {
    return launch_swapped_policy(
        a, b, sfa, sfb, m, n, k, alpha, device_index);
  }
  return launch_normal_policy(
      a, b, sfa, sfb, m, n, k, alpha, device_index);
#endif
}

at::Tensor gemm_cuda(at::Tensor const& a,
                     at::Tensor const& b,
                     at::Tensor const& sfa,
                     at::Tensor const& sfb,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     double alpha) {
  return gemm_cuda_impl(a, b, sfa, sfb, m, n, k, alpha);
}

}  // namespace

TORCH_LIBRARY(mxfp6, m) {
  m.def("gemm(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "int m, int n, int k, float alpha=1.0) -> Tensor");
  m.def("pack_fp6(Tensor codes) -> Tensor");
  m.def("unpack_fp6(Tensor packed, int rows, int k) -> Tensor");
  m.def("pack_scales(Tensor logical, int rows, int k) -> Tensor");
  m.def("unpack_scales(Tensor packed, int rows, int k) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxfp6, CUDA, m) {
  m.impl("gemm", &gemm_cuda);
  m.impl("pack_fp6", &mxfp6_gemm::torch_ext::pack_fp6_cuda);
  m.impl("unpack_fp6", &mxfp6_gemm::torch_ext::unpack_fp6_cuda);
  m.impl("pack_scales", &mxfp6_gemm::torch_ext::pack_scales_cuda);
  m.impl("unpack_scales", &mxfp6_gemm::torch_ext::unpack_scales_cuda);
}
