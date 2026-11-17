#include <c10/cuda/CUDACachingAllocator.h>
#include "cutlass/util/packed_stride.hpp"
#include "mxfp6_gemm/kernel_normal.hpp"
#include "mxfp6_gemm/kernel_swapped.hpp"
#include "mxfp6_gemm/packing.hpp"
#include "mxfp6_gemm/runtime.hpp"
#include "mxfp6_gemm/prepared.hpp"

using namespace cute;
namespace mxfp6_native {
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

template <class Gemm>
void configure_dynamic_smem(int device_index) {
  using GemmKernel = typename Gemm::GemmKernel;
  if constexpr (GemmKernel::SharedStorageSize >= (48 << 10)) {
    static std::mutex configured_devices_mutex;
    static std::set<int> configured_devices;
    std::lock_guard<std::mutex> lock(configured_devices_mutex);
    if (configured_devices.count(device_index) != 0) {
      return;
    }
    cudaError_t const result = cudaFuncSetAttribute(
        cutlass::device_kernel<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        GemmKernel::SharedStorageSize);
    TORCH_CHECK(
        result == cudaSuccess,
        "failed to configure Stream-K dynamic shared memory: ",
        cudaGetErrorString(result));
    configured_devices.insert(device_index);
  }
}

template <class Kernel, class Arguments>
void run_gemm(Arguments& arguments,
              at::Tensor const& workspace_anchor,
              int device_index,
              bool use_pdl) {
  using Gemm = typename Kernel::Gemm;
  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS can_implement failed: ", cutlassGetStatusString(status));

  WorkspaceLayout const workspace_layout =
      get_workspace_layout<Kernel>(arguments);
  size_t const workspace_bytes = Kernel::IsStreamK
      ? workspace_layout.reduction_bytes + workspace_layout.barrier_bytes
      : Gemm::get_workspace_size(arguments);
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  if constexpr (cutlass::sizeof_bits<typename Kernel::ElementA>::value == 8 ||
                cutlass::sizeof_bits<typename Kernel::ElementB>::value == 8) {
    if (preparing != nullptr) {
      // Own the arena independently of the serving workspace pool. initialize()
      // builds TMA descriptors and initializes barriers once, outside capture.
      auto prepared = std::make_shared<Gemm>();
      at::Tensor workspace;
      if (workspace_bytes)
        workspace = at::empty({static_cast<int64_t>(workspace_bytes)},
                              workspace_anchor.options().dtype(at::kByte));
      void* ptr = workspace.defined() ? workspace.data_ptr() : nullptr;
      status = prepared->initialize(arguments, ptr, stream.stream());
      TORCH_CHECK(status == cutlass::Status::kSuccess,
                  "CUTLASS prepare failed: ", cutlassGetStatusString(status));
      preparing->workspace = workspace;
      using Params = typename Gemm::Params;
      preparing->parameters = std::shared_ptr<void>(prepared,
          const_cast<Params*>(&prepared->params()));
      preparing->parameter_bytes = sizeof(Params);
      preparing->grid = Gemm::get_grid_shape(prepared->params());
      preparing->block = Gemm::GemmKernel::get_block_shape();
      preparing->shared_bytes = Gemm::GemmKernel::SharedStorageSize;
      preparing->reset_workspace = [workspace, workspace_layout](cudaStream_t stream) {
        if (workspace_layout.barrier_bytes && !workspace_layout.resettable) {
          auto* barriers = static_cast<char*>(workspace.data_ptr()) + workspace_layout.reduction_bytes;
          C10_CUDA_CHECK(cudaMemsetAsync(barriers, 0, workspace_layout.barrier_bytes, stream));
        }
      };
      preparing->run = [prepared](cudaStream_t stream, bool pdl) {
        auto status = prepared->run(stream, nullptr, pdl);
        TORCH_CHECK(status == cutlass::Status::kSuccess,
                    "CUTLASS prepared launch failed: ", cutlassGetStatusString(status));
      };
      return;
    }
  }
  WorkspaceSelection const workspace_selection = workspace_pool.select_workspace(
      device_index, stream.stream(), workspace_layout);

  if (workspace_selection.persistent) {
    configure_dynamic_smem<Gemm>(device_index);
    status = gemm.update(arguments, workspace_selection.pointer);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS update failed: ", cutlassGetStatusString(status));
    status = gemm.run(stream.stream(), nullptr, use_pdl);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS launch failed: ", cutlassGetStatusString(status));
    return;
  }

  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_bytes)},
        workspace_anchor.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }

  status = gemm.initialize(arguments, workspace_ptr, stream.stream());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS initialize failed: ", cutlassGetStatusString(status));
  status = gemm.run(stream.stream(), nullptr, use_pdl);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS launch failed: ", cutlassGetStatusString(status));
  if (workspace.defined()) {
    c10::cuda::CUDACachingAllocator::recordStream(
        workspace.storage().data_ptr(), stream);
  }
}

template <class Kernel>
at::Tensor launch_swapped_typed(at::Tensor const& a,
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
                          int splits = 1,
                          int sm_count = 0,
                          bool use_pdl = false) {
  using Gemm = typename Kernel::Gemm;
  using BlockScaledConfig = typename Kernel::BlockScaledConfig;
  static_assert(
      std::is_same_v<typename Kernel::ElementD, cutlass::half_t> ||
      std::is_same_v<typename Kernel::ElementD, cutlass::bfloat16_t>);
  constexpr at::ScalarType output_dtype =
      std::is_same_v<typename Kernel::ElementD, cutlass::bfloat16_t>
      ? at::kBFloat16
      : at::kHalf;
  auto output = at::empty({m, n}, a.options().dtype(output_dtype));
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
       reinterpret_cast<typename Kernel::ElementD*>(output.data_ptr()),
       stride_d}};

  arguments.hw_info.device_id = device_index;
  arguments.hw_info.sm_count = sm_count;
  configure_scheduler<Kernel>(
      arguments, max_swizzle_size, raster_order, splits);

  run_gemm<Kernel>(arguments, a, device_index, use_pdl);
  return output;
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
                          int splits = 1,
                          int sm_count = 0,
                          bool use_pdl = false) {
  if (current_output_dtype == at::kBFloat16) {
    using BFloat16Kernel = typename Kernel::template RebindOutput<
        cutlass::bfloat16_t>;
    return launch_swapped_typed<BFloat16Kernel>(
        a, b, sfa, sfb, m, n, k, alpha, device_index, max_swizzle_size,
        raster_order, splits, sm_count, use_pdl);
  }
  return launch_swapped_typed<Kernel>(
      a, b, sfa, sfb, m, n, k, alpha, device_index, max_swizzle_size,
      raster_order, splits, sm_count, use_pdl);
}

template <class Kernel>
at::Tensor launch_normal_typed(at::Tensor const& a,
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
                         int splits = 1,
                         int sm_count = 0,
                         bool use_pdl = false) {
  using Gemm = typename Kernel::Gemm;
  using BlockScaledConfig = typename Kernel::BlockScaledConfig;
  static_assert(
      std::is_same_v<typename Kernel::ElementD, cutlass::half_t> ||
      std::is_same_v<typename Kernel::ElementD, cutlass::bfloat16_t>);
  constexpr at::ScalarType output_dtype =
      std::is_same_v<typename Kernel::ElementD, cutlass::bfloat16_t>
      ? at::kBFloat16
      : at::kHalf;
  auto output = at::empty({m, n}, a.options().dtype(output_dtype));
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
       reinterpret_cast<typename Kernel::ElementD*>(output.data_ptr()),
       stride_d}};

  arguments.hw_info.device_id = device_index;
  arguments.hw_info.sm_count = sm_count;
  configure_scheduler<Kernel>(
      arguments, max_swizzle_size, raster_order, splits);

  run_gemm<Kernel>(arguments, a, device_index, use_pdl);
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
                         int splits = 1,
                         int sm_count = 0,
                         bool use_pdl = false) {
  if (current_output_dtype == at::kBFloat16) {
    using BFloat16Kernel = typename Kernel::template RebindOutput<
        cutlass::bfloat16_t>;
    return launch_normal_typed<BFloat16Kernel>(
        a, b, sfa, sfb, m, n, k, alpha, device_index, max_swizzle_size,
        raster_order, splits, sm_count, use_pdl);
  }
  return launch_normal_typed<Kernel>(
      a, b, sfa, sfb, m, n, k, alpha, device_index, max_swizzle_size,
      raster_order, splits, sm_count, use_pdl);
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

  if (!stream_k_enabled()) {
    if (m <= 16) {
      return launch_swapped<kernels::Kernel64x16x128Stage3Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index);
    }
    return launch_swapped<kernels::Kernel64x32x128Stage3Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index);
  }

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

  if (!stream_k_enabled()) {
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

enum class W6A8Kernel128 {
  Pingpong,
  Cooperative,
  StreamK,
  StaticPingpong,
  StaticCooperative,
};

at::Tensor launch_w6a8_128(at::Tensor const& a,
                           at::Tensor const& b,
                           at::Tensor const& sfa,
                           at::Tensor const& sfb,
                           int64_t m,
                           int64_t n,
                           int64_t k,
                           double alpha,
                           int device_index,
                           int sm_count,
                           W6A8Kernel128 kernel,
                           int swizzle,
                           RasterOrder raster,
                           bool use_pdl) {
  namespace kernels = mxfp6_gemm::normal;
  if (kernel == W6A8Kernel128::StreamK && !stream_k_enabled()) {
    kernel = W6A8Kernel128::Pingpong;
  }
  switch (kernel) {
    case W6A8Kernel128::Pingpong:
      return launch_normal<kernels::KernelW6A8_128x128x128Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle, raster, 1, 0, use_pdl);
    case W6A8Kernel128::Cooperative:
      return launch_normal<kernels::KernelW6A8_128x128x128Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle, raster, 1, 0, use_pdl);
    case W6A8Kernel128::StreamK:
      return launch_normal<kernels::KernelW6A8_128x128x128StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle, raster, 1, 0, use_pdl);
    case W6A8Kernel128::StaticPingpong:
      return launch_normal<kernels::KernelW6A8_128x128x128StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle, raster, 1, sm_count, use_pdl);
    case W6A8Kernel128::StaticCooperative:
      return launch_normal<kernels::KernelW6A8_128x128x128StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle, raster, 1, sm_count, use_pdl);
  }
  TORCH_CHECK(false, "unreachable W6A8 kernel selection");
}

// Stable candidate IDs used by the Python first-use autotuner. Full AOT keeps
// every candidate available; prepared execution can reuse the same arguments
// and owned CUTLASS Params with an explicitly selected DeepJIT kernel.
at::Tensor launch_w6a8_config(at::Tensor const& a,
                              at::Tensor const& b,
                              at::Tensor const& sfa,
                              at::Tensor const& sfb,
                              int64_t m,
                              int64_t n,
                              int64_t k,
                              double alpha,
                              int device_index,
                              int sm_count,
                              int64_t config_id,
                              int64_t swizzle,
                              int64_t raster_order,
                              bool use_pdl) {
  namespace normal = mxfp6_gemm::normal;
  namespace swapped = mxfp6_gemm::swapped;
  check_swizzle(swizzle);
  RasterOrder const raster = parse_raster_order(raster_order);
  int const swizzle_value = static_cast<int>(swizzle);

#include "mxfp6_gemm/w6a8_candidates.inc"
  TORCH_CHECK(false, "unknown W6A8 config_id ", config_id,
              "; expected an integer in [0, 29]");
}


at::Tensor launch_w6a6_policy(at::Tensor const& a, at::Tensor const& b, at::Tensor const& sfa, at::Tensor const& sfb, int64_t m, int64_t n, int64_t k, double alpha, int device_index, int sm_count) {
  bool const target_nk = is_target_nk(n, k);

  // Profiler-ranked mixed-MMA winners for Qwen3.5-27B TP2. The public tensors
  // remain packed W6A6. For large M, expanding A once is substantially cheaper
  // than converting every register fragment in the mainloop and exposes
  // SM120's fast E4M3 x E3M2 instruction path.
  if (m >= 512 && target_nk) {
    auto a_e4m3 = mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda(a, m, k);
    return launch_w6a8_policy(
        a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
        sm_count, true);
  }

  // The M=64/96 transition region needs explicit profiler-selected tiles:
  // the generic swapped policy underfills several target layers, while moving
  // every layer to mixed MMA would pay conversion where native W6A6 is faster.
  if (m == 64 && target_nk) {
    if (n == 8192 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x32StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          2, RasterOrder::AlongM, 1, sm_count);
    }
    if (n == 5120 && k == 3072) {
      auto a_e4m3 = mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda(a, m, k);
      return launch_normal<
          mxfp6_gemm::normal::KernelW6A8_64x64x128Stage4StaticPingpong>(
          a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, sm_count, true);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x32StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::AlongM, 1, sm_count);
    }
    if (n == 17408 && k == 5120) {
      return launch_normal<
          mxfp6_gemm::normal::TargetKernel64x64x128Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM);
    }
    auto a_e4m3 = mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda(a, m, k);
    return launch_normal<
        mxfp6_gemm::normal::KernelW6A8_64x64x128StaticPingpong>(
        a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
        8, RasterOrder::AlongM, 1, sm_count, true);
  }

  if (m == 96 && target_nk) {
    if ((n == 8192 && k == 5120) ||
        (n == 17408 && k == 5120)) {
      auto a_e4m3 = mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda(a, m, k);
      if (n == 8192) {
        return launch_normal<
            mxfp6_gemm::normal::KernelW6A8_64x128x128StaticPingpong>(
            a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
            2, RasterOrder::AlongM, 1, sm_count, true);
      }
      return launch_normal<
          mxfp6_gemm::normal::KernelW6A8_64x128x128Pingpong>(
          a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongM, 1, 0, true);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x32StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongN, 1, sm_count);
    }
    return launch_swapped<
        mxfp6_gemm::swapped::TargetKernel128x32Cooperative>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, n == 5120 && k == 8704
               ? RasterOrder::AlongN
               : RasterOrder::AlongM);
  }

  // At M=32 most target layers leave enough native-W6A6 tensor-core
  // efficiency on the table to repay expansion of the small activation. The
  // shallow-K output projection is the exception: retain native W6A6 and use
  // its profiler-selected static-persistent configuration there.
  if (m == 32 && target_nk) {
    if (n == 5120 && k == 3072) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel64x16x128Stage6StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::AlongM, 1, sm_count);
    }
    auto a_e4m3 = mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda(a, m, k);
    if (n == 8192 && k == 5120) {
      return launch_normal<
          mxfp6_gemm::normal::KernelW6A8_64x64x128StaticPingpong>(
          a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongN, 1, sm_count, true);
    }
    if (n == 7168 && k == 5120) {
      return launch_normal<
          mxfp6_gemm::normal::KernelW6A8_64x64x256StaticPingpong>(
          a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::AlongN, 1, sm_count, true);
    }
    if (n == 17408 && k == 5120) {
      return launch_normal<
          mxfp6_gemm::normal::KernelW6A8_64x64x128Pingpong>(
          a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, 0, true);
    }
    return launch_normal<
        mxfp6_gemm::normal::KernelW6A8_64x64x128StaticPingpong>(
        a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
        2, RasterOrder::AlongN, 1, sm_count, true);
  }

  if (target_nk && m == 1) {
    if (n == 8192 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8Stage4StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongM, 1, sm_count);
    }
    if (n == 5120 && k == 3072) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::AlongN, 1, sm_count);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongN, 1, sm_count);
    }
    if (n == 17408 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, sm_count);
    }
    return launch_swapped<
        mxfp6_gemm::swapped::TargetKernel64x16x256Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, RasterOrder::AlongM, 1, sm_count);
  }
  if (target_nk && m == 16) {
    if (n == 8192 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongN, 1, sm_count);
    }
    if (n == 5120 && k == 3072) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8Stage4StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongN, 1, sm_count);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, sm_count);
    }
    if (n == 17408 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel64x16x256Stage3StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongN, 1, sm_count);
    }
    return launch_swapped<
        mxfp6_gemm::swapped::TargetKernel64x16x512Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, RasterOrder::AlongM, 1, sm_count);
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
}
} // namespace mxfp6_native
