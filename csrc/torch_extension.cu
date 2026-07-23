#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "mxfp6_gemm/kernel_normal.hpp"
#include "mxfp6_gemm/kernel_swapped.hpp"
#include "mxfp6_gemm/packing.hpp"
#include "mxfp6_gemm/quantization.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

enum class RasterOrder {
  Heuristic,
  AlongM,
  AlongN,
};

struct W6A8ShapeKey {
  int device_index;
  int64_t m;
  int64_t n;
  int64_t k;
  at::ScalarType output_dtype;

  bool operator==(W6A8ShapeKey const& other) const {
    return device_index == other.device_index && m == other.m &&
        n == other.n && k == other.k && output_dtype == other.output_dtype;
  }
};

struct W6A8ShapeKeyHash {
  size_t operator()(W6A8ShapeKey const& key) const {
    size_t value = std::hash<int>{}(key.device_index);
    auto combine = [&value](int64_t field) {
      value ^= std::hash<int64_t>{}(field) + 0x9e3779b9 +
          (value << 6) + (value >> 2);
    };
    combine(key.m);
    combine(key.n);
    combine(key.k);
    combine(static_cast<int64_t>(key.output_dtype));
    return value;
  }
};

struct W6A8LaunchConfig {
  int64_t config_id;
  int64_t swizzle;
  int64_t raster_order;
};

std::shared_mutex w6a8_overrides_mutex;
std::unordered_map<W6A8ShapeKey, W6A8LaunchConfig, W6A8ShapeKeyHash>
    w6a8_overrides;

struct WorkspaceLayout {
  size_t reduction_bytes = 0;
  size_t barrier_bytes = 0;
  bool resettable = false;
};

enum class WorkspaceArenaMode {
  Disabled,
  Planning,
  Frozen,
};

struct WorkspaceLane {
  cudaStream_t stream = nullptr;
  at::Tensor storage;
};

struct WorkspaceArena {
  WorkspaceArenaMode mode = WorkspaceArenaMode::Disabled;
  size_t max_reduction_bytes = 0;
  size_t max_barrier_bytes = 0;
  std::set<std::tuple<size_t, size_t, bool>> layouts;
  std::vector<WorkspaceLane> lanes;
  int64_t persistent_launches = 0;
  int64_t fallback_launches = 0;
};

std::mutex workspace_arenas_mutex;
std::unordered_map<int, WorkspaceArena> workspace_arenas;
thread_local bool collect_workspace_layouts = true;

struct WorkspaceSelection {
  at::Tensor owner;
  void* pointer = nullptr;
  bool persistent = false;
};

bool find_w6a8_override(int device_index,
                        int64_t m,
                        int64_t n,
                        int64_t k,
                        at::ScalarType output_dtype,
                        W6A8LaunchConfig& result) {
  std::shared_lock<std::shared_mutex> lock(w6a8_overrides_mutex);
  auto const iterator = w6a8_overrides.find(
      {device_index, m, n, k, output_dtype});
  if (iterator == w6a8_overrides.end()) {
    return false;
  }
  result = iterator->second;
  return true;
}

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

template <class Kernel, class Arguments>
WorkspaceLayout get_workspace_layout(Arguments const& arguments) {
  if constexpr (!Kernel::IsStreamK) {
    return {};
  } else {
    using Gemm = typename Kernel::Gemm;
    using GemmKernel = typename Gemm::GemmKernel;
    using TileScheduler = typename GemmKernel::TileScheduler;
    constexpr uint32_t epilogue_subtiles =
        GemmKernel::CollectiveEpilogue::get_store_pipe_increment(
            typename GemmKernel::TileShape{});
    auto const scheduler_layout =
        TileScheduler::template get_workspace_layout<
            typename GemmKernel::ProblemShape,
            typename GemmKernel::ElementAccumulator>(
            arguments.scheduler,
            arguments.problem_shape,
            arguments.hw_info,
            GemmKernel::NumMmaWarpGroups,
            epilogue_subtiles,
            1);
    WorkspaceLayout const layout{
        scheduler_layout.reduction_bytes,
        scheduler_layout.barrier_bytes,
        scheduler_layout.resettable};
    size_t const total_bytes = Gemm::get_workspace_size(arguments);
    TORCH_CHECK(
        total_bytes == layout.reduction_bytes + layout.barrier_bytes,
        "Stream-K workspace layout mismatch: total=", total_bytes,
        ", reduction=", layout.reduction_bytes,
        ", barrier=", layout.barrier_bytes);
    return layout;
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

bool stream_is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaError_t const result = cudaStreamIsCapturing(stream, &capture_status);
  TORCH_CHECK(
      result == cudaSuccess,
      "cudaStreamIsCapturing failed: ", cudaGetErrorString(result));
  return capture_status != cudaStreamCaptureStatusNone;
}

WorkspaceSelection select_workspace(
    int device_index,
    cudaStream_t stream,
    WorkspaceLayout const& layout) {
  WorkspaceSelection selection;
  if (layout.reduction_bytes == 0 && layout.barrier_bytes == 0) {
    return selection;
  }
  std::string fallback_reason;
  {
    std::lock_guard<std::mutex> lock(workspace_arenas_mutex);
    auto iterator = workspace_arenas.find(device_index);
    if (iterator == workspace_arenas.end()) {
      return selection;
    }
    WorkspaceArena& arena = iterator->second;
    if (arena.mode == WorkspaceArenaMode::Planning) {
      if (collect_workspace_layouts) {
        arena.layouts.emplace(
            layout.reduction_bytes, layout.barrier_bytes, layout.resettable);
        arena.max_reduction_bytes =
            std::max(arena.max_reduction_bytes, layout.reduction_bytes);
        arena.max_barrier_bytes =
            std::max(arena.max_barrier_bytes, layout.barrier_bytes);
      }
      return selection;
    }
    if (arena.mode != WorkspaceArenaMode::Frozen ||
        layout.barrier_bytes == 0) {
      return selection;
    }
    if (!layout.resettable) {
      fallback_reason = "the selected Stream-K layout is not resettable";
    } else if (layout.reduction_bytes > arena.max_reduction_bytes ||
               layout.barrier_bytes > arena.max_barrier_bytes) {
      fallback_reason =
          "the selected Stream-K layout exceeds the frozen arena capacity";
    } else {
      auto lane = std::find_if(
          arena.lanes.begin(), arena.lanes.end(),
          [stream](WorkspaceLane const& candidate) {
            return candidate.stream == stream;
          });
      if (lane == arena.lanes.end()) {
        TORCH_CHECK(
            !stream_is_capturing(stream),
            "Stream-K persistent workspace has no lane for the current "
            "CUDA Graph capture stream. Run an eager warmup on this stream "
            "after finalize_workspace_planning() and before capture.");
        size_t const arena_bytes =
            arena.max_reduction_bytes + arena.max_barrier_bytes;
        auto options = at::TensorOptions()
            .device(at::Device(at::kCUDA, device_index))
            .dtype(at::kByte);
        at::Tensor storage = at::empty(
            {static_cast<int64_t>(arena_bytes)}, options);
        if (arena.max_barrier_bytes > 0) {
          auto* barrier_ptr =
              static_cast<uint8_t*>(storage.data_ptr()) +
              arena.max_reduction_bytes;
          cudaError_t const result = cudaMemsetAsync(
              barrier_ptr, 0, arena.max_barrier_bytes, stream);
          TORCH_CHECK(
              result == cudaSuccess,
              "failed to initialize persistent Stream-K lane barriers: ",
              cudaGetErrorString(result));
        }
        arena.lanes.push_back({stream, std::move(storage)});
        lane = std::prev(arena.lanes.end());
      }
      selection.owner = lane->storage;
      auto* arena_ptr = static_cast<uint8_t*>(lane->storage.data_ptr());
      selection.pointer = arena_ptr + arena.max_reduction_bytes -
          layout.reduction_bytes;
      selection.persistent = true;
      ++arena.persistent_launches;
      return selection;
    }
    ++arena.fallback_launches;
  }

  TORCH_CHECK(
      !stream_is_capturing(stream),
      "Stream-K persistent workspace is unavailable during CUDA Graph "
      "capture: ", fallback_reason,
      ". Re-run workspace planning with every captured shape/config on "
      "the capture stream.");
  return selection;
}

int64_t round_up(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

bool is_target_nk(int64_t n, int64_t k) {
  return (n == 8192 && k == 5120) ||
      (n == 5120 && k == 3072) ||
      (n == 7168 && k == 5120) ||
      (n == 17408 && k == 5120) ||
      (n == 5120 && k == 8704);
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

at::ScalarType resolve_output_dtype(
    std::optional<at::ScalarType> output_dtype) {
  at::ScalarType const resolved = output_dtype.value_or(at::kHalf);
  TORCH_CHECK(resolved == at::kHalf || resolved == at::kBFloat16,
              "out_dtype must be torch.float16 or torch.bfloat16; got ",
              resolved);
  return resolved;
}

// Kernel selection is synchronous on the calling host thread. A scoped
// thread-local keeps output dtype out of the large policy call graph while
// remaining safe for concurrent launches from independent Python threads.
thread_local at::ScalarType current_output_dtype = at::kHalf;

class OutputDtypeGuard {
 public:
  explicit OutputDtypeGuard(at::ScalarType output_dtype)
      : previous_(current_output_dtype) {
    current_output_dtype = output_dtype;
  }

  ~OutputDtypeGuard() {
    current_output_dtype = previous_;
  }

 private:
  at::ScalarType previous_;
};

class WorkspaceCollectionGuard {
 public:
  explicit WorkspaceCollectionGuard(bool enabled)
      : previous_(collect_workspace_layouts) {
    collect_workspace_layouts = enabled;
  }

  ~WorkspaceCollectionGuard() {
    collect_workspace_layouts = previous_;
  }

 private:
  bool previous_;
};

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
  WorkspaceSelection const workspace_selection = select_workspace(
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

enum class W6A8Kernel128 {
  Pingpong,
  Cooperative,
  StreamK,
  StaticPingpong,
  StaticCooperative,
};

RasterOrder parse_raster_order(int64_t value) {
  switch (value) {
    case 0:
      return RasterOrder::Heuristic;
    case 1:
      return RasterOrder::AlongM;
    case 2:
      return RasterOrder::AlongN;
  }
  TORCH_CHECK(false, "raster_order must be 0 (heuristic), 1 (along_m), or "
                     "2 (along_n); got ", value);
}

void check_swizzle(int64_t value) {
  TORCH_CHECK(value == 1 || value == 2 || value == 4 || value == 8,
              "swizzle must be one of 1, 2, 4, or 8; got ", value);
}

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

// Stable candidate IDs used by the Python first-use autotuner. All candidates
// are native kernels compiled into this extension; JIT here means selecting a
// precompiled implementation for a new shape and caching that decision, not
// invoking the optional Humming backend.
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

  switch (config_id) {
    case 0:
      return launch_swapped<swapped::KernelW6A8_128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 1:
      return launch_swapped<
          swapped::KernelW6A8_128x8Stage4StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 2:
      return launch_swapped<
          swapped::KernelW6A8_128x8Stage5StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 3:
      return launch_swapped<swapped::KernelW6A8_128x8StreamK>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 4:
      return launch_swapped<
          swapped::KernelW6A8_128x16Stage4StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 5:
      return launch_swapped<swapped::KernelW6A8_64x16x256StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 6:
      return launch_normal<normal::KernelW6A8_64x64x128Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, 0, use_pdl);
    case 7:
      return launch_normal<normal::KernelW6A8_64x64x128StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 8:
      return launch_normal<
          normal::KernelW6A8_64x64x128Stage4StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 9:
      return launch_normal<normal::KernelW6A8_64x64x256StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 10:
      return launch_normal<normal::KernelW6A8_64x128x128Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, 0, use_pdl);
    case 11:
      return launch_normal<normal::KernelW6A8_64x128x128StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 12:
      return launch_w6a8_128(
          a, b, sfa, sfb, m, n, k, alpha, device_index, sm_count,
          W6A8Kernel128::Pingpong, swizzle_value, raster, use_pdl);
    case 13:
      return launch_w6a8_128(
          a, b, sfa, sfb, m, n, k, alpha, device_index, sm_count,
          W6A8Kernel128::Cooperative, swizzle_value, raster, use_pdl);
    case 14:
      return launch_w6a8_128(
          a, b, sfa, sfb, m, n, k, alpha, device_index, sm_count,
          W6A8Kernel128::StaticPingpong, swizzle_value, raster, use_pdl);
    case 15:
      return launch_w6a8_128(
          a, b, sfa, sfb, m, n, k, alpha, device_index, sm_count,
          W6A8Kernel128::StaticCooperative, swizzle_value, raster, use_pdl);
    case 16:
      return launch_w6a8_128(
          a, b, sfa, sfb, m, n, k, alpha, device_index, sm_count,
          W6A8Kernel128::StreamK, swizzle_value, raster, use_pdl);
    case 17:
      return launch_swapped<swapped::KernelW6A8_64x32x128Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, 0, use_pdl);
    case 18:
      return launch_swapped<swapped::KernelW6A8_64x32x128Stage3Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, 0, use_pdl);
    case 19:
      return launch_swapped<swapped::KernelW6A8_64x32x128StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 20:
      return launch_swapped<
          swapped::KernelW6A8_64x32x128Stage3StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 21:
      return launch_swapped<swapped::KernelW6A8_64x32x256Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, 0, use_pdl);
    case 22:
      return launch_swapped<swapped::KernelW6A8_64x32x256Stage2Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, 0, use_pdl);
    case 23:
      return launch_swapped<swapped::KernelW6A8_64x32x256StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 24:
      return launch_swapped<
          swapped::KernelW6A8_64x32x256Stage2StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 25:
      return launch_swapped<swapped::KernelW6A8_128x32Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, 0, use_pdl);
    case 26:
      return launch_swapped<swapped::KernelW6A8_128x32Stage2Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, 0, use_pdl);
    case 27:
      return launch_swapped<swapped::KernelW6A8_128x32StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
    case 28:
      return launch_swapped<
          swapped::KernelW6A8_128x32Stage2StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          swizzle_value, raster, 1, sm_count, use_pdl);
  }
  TORCH_CHECK(false, "unknown W6A8 config_id ", config_id,
              "; expected an integer in [0, 28]");
}

bool set_w6a8_config_cuda(at::Tensor const& anchor,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          int64_t config_id,
                          int64_t swizzle,
                          int64_t raster_order,
                          std::optional<at::ScalarType> output_dtype) {
  TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
  TORCH_CHECK(m > 0, "m must be positive; got ", m);
  TORCH_CHECK(n > 0 && n % 8 == 0,
              "n must be a positive multiple of 8; got ", n);
  TORCH_CHECK(k > 0 && k % 128 == 0,
              "k must be a positive multiple of 128; got ", k);
  W6A8ShapeKey const key{
      anchor.get_device(), m, n, k, resolve_output_dtype(output_dtype)};
  std::unique_lock<std::shared_mutex> lock(w6a8_overrides_mutex);
  if (config_id < 0) {
    w6a8_overrides.erase(key);
    return true;
  }
  TORCH_CHECK(config_id <= 28,
              "config_id must be in [0, 28], or negative to erase; got ",
              config_id);
  check_swizzle(swizzle);
  parse_raster_order(raster_order);
  w6a8_overrides[key] = {config_id, swizzle, raster_order};
  return true;
}

std::string w6a8_config_abi_cuda(at::Tensor const& anchor) {
  TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
  return "native-w6a8-29-v4";
}

// Native W6A8 dispatcher. The persistent B operand remains packed E3M2 while
// transient A is byte-addressable E4M3. Exact entries are profiler winners;
// the fallback uses Stream-K only for a visibly under-filled final wave.
at::Tensor launch_w6a8_policy(at::Tensor const& a,
                              at::Tensor const& b,
                              at::Tensor const& sfa,
                              at::Tensor const& sfb,
                              int64_t m,
                              int64_t n,
                              int64_t k,
                              double alpha,
                              int device_index,
                              int sm_count,
                              bool use_pdl) {
  namespace normal = mxfp6_gemm::normal;
  namespace swapped = mxfp6_gemm::swapped;
  bool const target_nk = is_target_nk(n, k);

  W6A8LaunchConfig override_config{};
  if (find_w6a8_override(
          device_index, m, n, k, current_output_dtype, override_config)) {
    return launch_w6a8_config(
        a, b, sfa, sfb, m, n, k, alpha, device_index, sm_count,
        override_config.config_id, override_config.swizzle,
        override_config.raster_order, use_pdl);
  }

  // Keeping logical M in the tensor-core N dimension avoids wasting a 64- or
  // 128-row activation tile in decode and very-small-prefill workloads.
  if (target_nk && m == 1) {
    if (n == 8192 && k == 5120) {
      return launch_swapped<swapped::KernelW6A8_128x8Stage4StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::Heuristic, 1, sm_count, use_pdl);
    }
    if (n == 5120 && k == 3072) {
      return launch_swapped<swapped::KernelW6A8_128x8Stage5StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, sm_count, use_pdl);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<swapped::KernelW6A8_128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::Heuristic, 1, sm_count, use_pdl);
    }
    if (n == 17408 && k == 5120) {
      return launch_swapped<swapped::KernelW6A8_128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, sm_count, use_pdl);
    }
    return launch_swapped<swapped::KernelW6A8_128x8StreamK>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        2, RasterOrder::AlongM, 1, sm_count, use_pdl);
  }

  if (target_nk && m == 16) {
    if (n == 8192 && k == 5120) {
      return launch_swapped<swapped::KernelW6A8_64x16x256StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          2, RasterOrder::AlongM, 1, sm_count, use_pdl);
    }
    if (n == 5120 && k == 3072) {
      return launch_swapped<swapped::KernelW6A8_64x16x256StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          2, RasterOrder::AlongM, 1, sm_count, use_pdl);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<swapped::KernelW6A8_64x16x256StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongM, 1, sm_count, use_pdl);
    }
    if (n == 17408 && k == 5120) {
      return launch_swapped<swapped::KernelW6A8_128x16Stage4StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, sm_count, use_pdl);
    }
    return launch_swapped<swapped::KernelW6A8_64x16x256StaticPingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        2, RasterOrder::AlongM, 1, sm_count, use_pdl);
  }

  if (target_nk && m == 32) {
    if (n == 8192 && k == 5120) {
      return launch_swapped<swapped::KernelW6A8_128x32Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, 0, use_pdl);
    }
    if (n == 5120 && k == 3072) {
      return launch_swapped<swapped::KernelW6A8_64x32x256StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          2, RasterOrder::AlongM, 1, sm_count, use_pdl);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<swapped::KernelW6A8_128x32Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, 0, use_pdl);
    }
    if (n == 17408 && k == 5120) {
      return launch_swapped<swapped::KernelW6A8_128x32Cooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::Heuristic, 1, 0, use_pdl);
    }
    return launch_swapped<swapped::KernelW6A8_64x32x128Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, RasterOrder::Heuristic, 1, 0, use_pdl);
  }

  if (target_nk && m == 64) {
    if (n == 5120 && k == 3072) {
      return launch_normal<normal::KernelW6A8_64x64x128Stage4StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, sm_count, use_pdl);
    }
    return launch_normal<normal::KernelW6A8_64x64x128StaticPingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        n == 5120 && k == 8704 ? 8 : 2,
        RasterOrder::AlongM, 1, sm_count, use_pdl);
  }

  if (target_nk && m == 96) {
    if (n == 17408 && k == 5120) {
      return launch_normal<normal::KernelW6A8_64x128x128Pingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongM, 1, 0, use_pdl);
    }
    return launch_normal<normal::KernelW6A8_64x128x128StaticPingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        n == 8192 ? 2 : 1, RasterOrder::AlongM, 1, sm_count, use_pdl);
  }

  if (target_nk && m == 512) {
    if (n == 8192 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::Pingpong,
          4, RasterOrder::AlongM, use_pdl);
    }
    if (n == 5120 && k == 3072) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::StaticPingpong,
          8, RasterOrder::AlongM, use_pdl);
    }
    if (n == 7168 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::StreamK,
          1, RasterOrder::AlongN, use_pdl);
    }
    if (n == 17408 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::StreamK,
          8, RasterOrder::AlongN, use_pdl);
    }
    return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
        device_index, sm_count, W6A8Kernel128::Pingpong,
        1, RasterOrder::AlongM, use_pdl);
  }

  if (target_nk && m == 1024) {
    if (n == 8192 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::StreamK,
          1, RasterOrder::AlongM, use_pdl);
    }
    if (n == 7168 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::Pingpong,
          4, RasterOrder::AlongM, use_pdl);
    }
    if (n == 17408 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::StreamK,
          1, RasterOrder::AlongM, use_pdl);
    }
    return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
        device_index, sm_count, W6A8Kernel128::Pingpong,
        1, RasterOrder::AlongM, use_pdl);
  }

  if (target_nk && m == 2048) {
    if ((n == 8192 && k == 5120) || (n == 7168 && k == 5120)) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::StreamK,
          n == 7168 ? 2 : 1, RasterOrder::AlongM, use_pdl);
    }
    if (n == 17408 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::Pingpong,
          1, RasterOrder::AlongN, use_pdl);
    }
    return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
        device_index, sm_count, W6A8Kernel128::Cooperative,
        1, RasterOrder::AlongM, use_pdl);
  }

  if (target_nk && m == 4096) {
    if (n == 8192 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::StaticCooperative,
          1, RasterOrder::AlongM, use_pdl);
    }
    if (n == 5120 && k == 3072) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::StaticPingpong,
          2, RasterOrder::AlongM, use_pdl);
    }
    if (n == 7168 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::Cooperative,
          1, RasterOrder::AlongM, use_pdl);
    }
    if (n == 17408 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::Pingpong,
          1, RasterOrder::AlongN, use_pdl);
    }
    return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
        device_index, sm_count, W6A8Kernel128::StaticPingpong,
        1, RasterOrder::AlongM, use_pdl);
  }

  if (target_nk && m == 8192) {
    if (n == 7168 && k == 5120) {
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::Pingpong,
          1, RasterOrder::AlongN, use_pdl);
    }
    if (n == 5120 && k == 8704) {
      if (current_output_dtype == at::kHalf) {
        return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
            device_index, sm_count, W6A8Kernel128::Pingpong,
            1, RasterOrder::AlongM, use_pdl);
      }
      return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
          device_index, sm_count, W6A8Kernel128::StreamK,
          1, RasterOrder::AlongN, use_pdl);
    }
    return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
        device_index, sm_count, W6A8Kernel128::StaticPingpong,
        1, (n == 5120 && k == 3072)
               ? RasterOrder::AlongM
               : RasterOrder::AlongN,
        use_pdl);
  }

  if (m <= 8) {
    return launch_swapped<swapped::KernelW6A8_128x8Stage4StaticCooperative>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        4, RasterOrder::Heuristic, 1, sm_count, use_pdl);
  }
  if (m <= 16) {
    return launch_swapped<swapped::KernelW6A8_128x8StaticCooperative>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        2, RasterOrder::Heuristic, 1, sm_count, use_pdl);
  }
  if (m <= 64) {
    return launch_normal<normal::KernelW6A8_64x64x128StaticPingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        2, RasterOrder::Heuristic, 1, sm_count, use_pdl);
  }
  if (m <= 128) {
    return launch_normal<normal::KernelW6A8_64x128x128StaticPingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        2, RasterOrder::Heuristic, 1, sm_count, use_pdl);
  }

  int64_t const tiles = round_up(m, 128) / 128 * (round_up(n, 128) / 128);
  int64_t const tail = tiles % sm_count;
  bool const underfilled_tail = k >= 2048 &&
      ((tiles < sm_count) ||
       (tiles <= 4LL * sm_count && tail > 0 && tail < sm_count / 2));
  return launch_w6a8_128(a, b, sfa, sfb, m, n, k, alpha,
      device_index, sm_count,
      underfilled_tail ? W6A8Kernel128::StreamK
                       : W6A8Kernel128::Pingpong,
      underfilled_tail ? 1 : 2, RasterOrder::Heuristic, use_pdl);
}

at::Tensor gemm_cuda_impl(at::Tensor const& a,
                          at::Tensor const& b,
                          at::Tensor const& sfa,
                          at::Tensor const& sfb,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          double alpha,
                          at::ScalarType output_dtype) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  TORCH_CHECK(false, "mxfp6_torch must be compiled for sm_120a");
#else
  OutputDtypeGuard output_guard(output_dtype);
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

  int device_index = a.get_device();
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(device_index);
  TORCH_CHECK(properties.major == 12 && properties.minor == 0,
              "mxfp6::gemm requires SM120; current device is SM",
              properties.major, properties.minor);

  bool const target_nk = is_target_nk(n, k);

  // Profiler-ranked mixed-MMA winners for Qwen3.5-27B TP2. The public tensors
  // remain packed W6A6. For large M, expanding A once is substantially cheaper
  // than converting every register fragment in the mainloop and exposes
  // SM120's fast E4M3 x E3M2 instruction path.
  if (m >= 512 && target_nk) {
    auto a_e4m3 = mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda(a, m, k);
    return launch_w6a8_policy(
        a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
        properties.multiProcessorCount, true);
  }

  // The M=64/96 transition region needs explicit profiler-selected tiles:
  // the generic swapped policy underfills several target layers, while moving
  // every layer to mixed MMA would pay conversion where native W6A6 is faster.
  if (m == 64 && target_nk) {
    if (n == 8192 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x32StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          2, RasterOrder::AlongM, 1, properties.multiProcessorCount);
    }
    if (n == 5120 && k == 3072) {
      auto a_e4m3 = mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda(a, m, k);
      return launch_normal<
          mxfp6_gemm::normal::KernelW6A8_64x64x128Stage4StaticPingpong>(
          a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, properties.multiProcessorCount, true);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x32StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::AlongM, 1, properties.multiProcessorCount);
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
        8, RasterOrder::AlongM, 1, properties.multiProcessorCount, true);
  }

  if (m == 96 && target_nk) {
    if ((n == 8192 && k == 5120) ||
        (n == 17408 && k == 5120)) {
      auto a_e4m3 = mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda(a, m, k);
      if (n == 8192) {
        return launch_normal<
            mxfp6_gemm::normal::KernelW6A8_64x128x128StaticPingpong>(
            a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
            2, RasterOrder::AlongM, 1, properties.multiProcessorCount, true);
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
          1, RasterOrder::AlongN, 1, properties.multiProcessorCount);
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
          8, RasterOrder::AlongM, 1, properties.multiProcessorCount);
    }
    auto a_e4m3 = mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda(a, m, k);
    if (n == 8192 && k == 5120) {
      return launch_normal<
          mxfp6_gemm::normal::KernelW6A8_64x64x128StaticPingpong>(
          a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongN, 1, properties.multiProcessorCount, true);
    }
    if (n == 7168 && k == 5120) {
      return launch_normal<
          mxfp6_gemm::normal::KernelW6A8_64x64x256StaticPingpong>(
          a_e4m3, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::AlongN, 1, properties.multiProcessorCount, true);
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
        2, RasterOrder::AlongN, 1, properties.multiProcessorCount, true);
  }

  if (target_nk && m == 1) {
    if (n == 8192 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8Stage4StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongM, 1, properties.multiProcessorCount);
    }
    if (n == 5120 && k == 3072) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          8, RasterOrder::AlongN, 1, properties.multiProcessorCount);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongN, 1, properties.multiProcessorCount);
    }
    if (n == 17408 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, properties.multiProcessorCount);
    }
    return launch_swapped<
        mxfp6_gemm::swapped::TargetKernel64x16x256Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, RasterOrder::AlongM, 1, properties.multiProcessorCount);
  }
  if (target_nk && m == 16) {
    if (n == 8192 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongN, 1, properties.multiProcessorCount);
    }
    if (n == 5120 && k == 3072) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8Stage4StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongN, 1, properties.multiProcessorCount);
    }
    if (n == 7168 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel128x8StaticCooperative>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          1, RasterOrder::AlongM, 1, properties.multiProcessorCount);
    }
    if (n == 17408 && k == 5120) {
      return launch_swapped<
          mxfp6_gemm::swapped::TargetKernel64x16x256Stage3StaticPingpong>(
          a, b, sfa, sfb, m, n, k, alpha, device_index,
          4, RasterOrder::AlongN, 1, properties.multiProcessorCount);
    }
    return launch_swapped<
        mxfp6_gemm::swapped::TargetKernel64x16x512Pingpong>(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        1, RasterOrder::AlongM, 1, properties.multiProcessorCount);
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
                     double alpha,
                     std::optional<at::ScalarType> output_dtype) {
  return gemm_cuda_impl(
      a, b, sfa, sfb, m, n, k, alpha,
      resolve_output_dtype(output_dtype));
}

at::Tensor gemm_w6a8_cuda_impl(at::Tensor const& a,
                               at::Tensor const& b,
                               at::Tensor const& sfa,
                               at::Tensor const& sfb,
                               int64_t m,
                               int64_t n,
                               int64_t k,
                               double alpha,
                               int64_t config_id,
                               int64_t swizzle,
                               int64_t raster_order,
                               at::ScalarType output_dtype) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  TORCH_CHECK(false, "mxfp6_torch must be compiled for sm_120a");
#else
  OutputDtypeGuard output_guard(output_dtype);
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
              "operand size exceeds int64 range");

  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(a.device());
  check_input(a, "a", a.device());
  check_input(b, "b", a.device());
  check_input(sfa, "sfa", a.device());
  check_input(sfb, "sfb", a.device());
  TORCH_CHECK(a.numel() == m * k,
              "a must contain exactly ", m * k,
              " E4M3 bytes; got ", a.numel());
  TORCH_CHECK(b.numel() == n * k * 3 / 4,
              "b must contain exactly ", n * k * 3 / 4,
              " packed E3M2 bytes; got ", b.numel());
  TORCH_CHECK(sfa.numel() == round_up(m, 128) * k / 32,
              "sfa has an invalid physical-layout size");
  TORCH_CHECK(sfb.numel() == round_up(n, 128) * k / 32,
              "sfb has an invalid physical-layout size");

  int const device_index = a.get_device();
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(device_index);
  TORCH_CHECK(properties.major == 12 && properties.minor == 0,
              "mxfp6::gemm_w6a8 requires SM120; current device is SM",
              properties.major, properties.minor);
  if (config_id >= 0) {
    return launch_w6a8_config(
        a, b, sfa, sfb, m, n, k, alpha, device_index,
        properties.multiProcessorCount, config_id, swizzle, raster_order,
        false);
  }
  return launch_w6a8_policy(
      a, b, sfa, sfb, m, n, k, alpha, device_index,
      properties.multiProcessorCount, false);
#endif
}

at::Tensor gemm_w6a8_cuda(at::Tensor const& a,
                          at::Tensor const& b,
                          at::Tensor const& sfa,
                          at::Tensor const& sfb,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          double alpha,
                          std::optional<at::ScalarType> output_dtype) {
  return gemm_w6a8_cuda_impl(
      a, b, sfa, sfb, m, n, k, alpha, -1, 1, 0,
      resolve_output_dtype(output_dtype));
}

at::Tensor gemm_w6a8_config_cuda(at::Tensor const& a,
                                 at::Tensor const& b,
                                 at::Tensor const& sfa,
                                 at::Tensor const& sfb,
                                 int64_t m,
                                 int64_t n,
                                 int64_t k,
                                 double alpha,
                                 int64_t config_id,
                                 int64_t swizzle,
                                 int64_t raster_order,
                                 std::optional<at::ScalarType> output_dtype) {
  TORCH_CHECK(config_id >= 0,
              "config_id must be nonnegative; got ", config_id);
  // Explicit configs are candidate probes used by the autotuner. Only the
  // selected production dispatch should contribute to arena capacity.
  WorkspaceCollectionGuard collection_guard(false);
  return gemm_w6a8_cuda_impl(
      a, b, sfa, sfb, m, n, k, alpha,
      config_id, swizzle, raster_order, resolve_output_dtype(output_dtype));
}

at::Tensor gemm_from_float_cuda_impl(at::Tensor const& input,
                                     at::Tensor const& b,
                                     at::Tensor const& sfb,
                                     int64_t n,
                                     double alpha,
                                     int64_t config_id,
                                     int64_t swizzle,
                                     int64_t raster_order,
                                     at::ScalarType output_dtype) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  TORCH_CHECK(false, "mxfp6_torch must be compiled for sm_120a");
#else
  OutputDtypeGuard output_guard(output_dtype);
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must have shape [M,K]");
  int64_t const m = input.size(0);
  int64_t const k = input.size(1);
  TORCH_CHECK(m > 0, "M must be positive; got ", m);
  TORCH_CHECK(n > 0 && n % 8 == 0,
              "N must be a positive multiple of 8; got ", n);
  TORCH_CHECK(k > 0 && k % 128 == 0,
              "K must be a positive multiple of 128; got ", k);
  TORCH_CHECK(m <= std::numeric_limits<int>::max() &&
                  n <= std::numeric_limits<int>::max() &&
                  k <= std::numeric_limits<int>::max(),
              "M, N, and K must fit in a 32-bit integer");
  TORCH_CHECK(n <= std::numeric_limits<int64_t>::max() / k,
              "weight size exceeds int64 range");

  c10::cuda::CUDAGuard device_guard(input.device());
  check_input(b, "b", input.device());
  check_input(sfb, "sfb", input.device());
  TORCH_CHECK(b.numel() == n * k * 3 / 4,
              "b must contain exactly ", n * k * 3 / 4,
              " packed E3M2 bytes; got ", b.numel());
  TORCH_CHECK(sfb.numel() == round_up(n, 128) * k / 32,
              "sfb has an invalid physical-layout size");

  int const device_index = input.get_device();
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(device_index);
  TORCH_CHECK(properties.major == 12 && properties.minor == 0,
              "mxfp6::gemm_from_float requires SM120; current device is SM",
              properties.major, properties.minor);

  auto quantized = mxfp6_gemm::torch_ext::quantize_mxfp8_cuda(input);
  if (config_id >= 0) {
    return launch_w6a8_config(
        std::get<0>(quantized), b, std::get<1>(quantized), sfb,
        m, n, k, alpha, device_index, properties.multiProcessorCount,
        config_id, swizzle, raster_order, true);
  }
  return launch_w6a8_policy(
      std::get<0>(quantized), b, std::get<1>(quantized), sfb,
      m, n, k, alpha, device_index, properties.multiProcessorCount, true);
#endif
}

at::Tensor gemm_from_float_cuda(at::Tensor const& input,
                                at::Tensor const& b,
                                at::Tensor const& sfb,
                                int64_t n,
                                double alpha,
                                std::optional<at::ScalarType> output_dtype) {
  return gemm_from_float_cuda_impl(
      input, b, sfb, n, alpha, -1, 1, 0,
      resolve_output_dtype(output_dtype));
}

at::Tensor gemm_from_float_config_cuda(at::Tensor const& input,
                                       at::Tensor const& b,
                                       at::Tensor const& sfb,
                                       int64_t n,
                                       double alpha,
                                       int64_t config_id,
                                       int64_t swizzle,
                                       int64_t raster_order,
                                       std::optional<at::ScalarType> output_dtype) {
  TORCH_CHECK(config_id >= 0,
              "config_id must be nonnegative; got ", config_id);
  WorkspaceCollectionGuard collection_guard(false);
  return gemm_from_float_cuda_impl(
      input, b, sfb, n, alpha, config_id, swizzle, raster_order,
      resolve_output_dtype(output_dtype));
}

void check_workspace_anchor(at::Tensor const& anchor) {
  TORCH_CHECK(anchor.is_cuda(), "workspace anchor must be a CUDA tensor");
  cudaDeviceProp const& properties =
      *at::cuda::getDeviceProperties(anchor.get_device());
  TORCH_CHECK(
      properties.major == 12 && properties.minor == 0,
      "persistent Stream-K workspace requires SM120; current device is SM",
      properties.major, properties.minor);
}

c10::Dict<std::string, int64_t> workspace_stats_locked(
    WorkspaceArena const& arena) {
  c10::Dict<std::string, int64_t> stats;
  stats.insert("layouts", static_cast<int64_t>(arena.layouts.size()));
  stats.insert(
      "max_reduction_bytes",
      static_cast<int64_t>(arena.max_reduction_bytes));
  stats.insert(
      "max_barrier_bytes",
      static_cast<int64_t>(arena.max_barrier_bytes));
  stats.insert(
      "arena_bytes",
      static_cast<int64_t>(
          arena.max_reduction_bytes + arena.max_barrier_bytes));
  stats.insert("lanes", static_cast<int64_t>(arena.lanes.size()));
  stats.insert(
      "total_arena_bytes",
      static_cast<int64_t>(
          (arena.max_reduction_bytes + arena.max_barrier_bytes) *
          arena.lanes.size()));
  stats.insert(
      "planning", arena.mode == WorkspaceArenaMode::Planning ? 1 : 0);
  stats.insert("frozen", arena.mode == WorkspaceArenaMode::Frozen ? 1 : 0);
  stats.insert("persistent_launches", arena.persistent_launches);
  stats.insert("fallback_launches", arena.fallback_launches);
  return stats;
}

void begin_workspace_planning_cuda(at::Tensor const& anchor) {
  check_workspace_anchor(anchor);
  c10::cuda::CUDAGuard device_guard(anchor.device());
  int const device_index = anchor.get_device();
  std::lock_guard<std::mutex> lock(workspace_arenas_mutex);
  WorkspaceArena& arena = workspace_arenas[device_index];
  TORCH_CHECK(
      arena.mode != WorkspaceArenaMode::Frozen,
      "persistent Stream-K workspace for CUDA device ", device_index,
      " is already frozen; resizing it could invalidate captured graphs");
  TORCH_CHECK(
      arena.mode != WorkspaceArenaMode::Planning,
      "workspace planning is already active for CUDA device ",
      device_index);
  arena = WorkspaceArena{};
  arena.mode = WorkspaceArenaMode::Planning;
}

c10::Dict<std::string, int64_t> finalize_workspace_planning_cuda(
    at::Tensor const& anchor) {
  check_workspace_anchor(anchor);
  c10::cuda::CUDAGuard device_guard(anchor.device());
  int const device_index = anchor.get_device();
  auto stream = c10::cuda::getCurrentCUDAStream(device_index);
  std::lock_guard<std::mutex> lock(workspace_arenas_mutex);
  auto iterator = workspace_arenas.find(device_index);
  TORCH_CHECK(
      iterator != workspace_arenas.end() &&
          iterator->second.mode == WorkspaceArenaMode::Planning,
      "workspace planning is not active for CUDA device ", device_index);
  WorkspaceArena& arena = iterator->second;
  size_t const arena_bytes =
      arena.max_reduction_bytes + arena.max_barrier_bytes;
  TORCH_CHECK(
      arena_bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "persistent Stream-K workspace exceeds the tensor size limit");
  if (arena_bytes > 0) {
    at::Tensor storage = at::empty(
        {static_cast<int64_t>(arena_bytes)},
        anchor.options().dtype(at::kByte));
    if (arena.max_barrier_bytes > 0) {
      auto* barrier_ptr = static_cast<uint8_t*>(storage.data_ptr()) +
          arena.max_reduction_bytes;
      cudaError_t const result = cudaMemsetAsync(
          barrier_ptr, 0, arena.max_barrier_bytes, stream.stream());
      TORCH_CHECK(
          result == cudaSuccess,
          "failed to initialize persistent Stream-K barriers: ",
          cudaGetErrorString(result));
    }
    arena.lanes.push_back({stream.stream(), std::move(storage)});
  }
  arena.mode = WorkspaceArenaMode::Frozen;
  return workspace_stats_locked(arena);
}

c10::Dict<std::string, int64_t> workspace_stats_cuda(
    at::Tensor const& anchor) {
  check_workspace_anchor(anchor);
  std::lock_guard<std::mutex> lock(workspace_arenas_mutex);
  auto const iterator = workspace_arenas.find(anchor.get_device());
  if (iterator == workspace_arenas.end()) {
    return workspace_stats_locked(WorkspaceArena{});
  }
  return workspace_stats_locked(iterator->second);
}

bool workspace_barriers_zero_cuda(at::Tensor const& anchor) {
  check_workspace_anchor(anchor);
  c10::cuda::CUDAGuard device_guard(anchor.device());
  int const device_index = anchor.get_device();
  std::vector<at::Tensor> lane_storage;
  size_t barrier_offset = 0;
  size_t barrier_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(workspace_arenas_mutex);
    auto const iterator = workspace_arenas.find(device_index);
    TORCH_CHECK(
        iterator != workspace_arenas.end() &&
            iterator->second.mode == WorkspaceArenaMode::Frozen,
        "persistent workspace is not frozen for CUDA device ",
        device_index);
    WorkspaceArena const& arena = iterator->second;
    lane_storage.reserve(arena.lanes.size());
    for (WorkspaceLane const& lane : arena.lanes) {
      lane_storage.push_back(lane.storage);
    }
    barrier_offset = arena.max_reduction_bytes;
    barrier_bytes = arena.max_barrier_bytes;
  }
  if (barrier_bytes == 0) {
    return true;
  }
  cudaError_t result = cudaDeviceSynchronize();
  TORCH_CHECK(
      result == cudaSuccess,
      "failed to synchronize persistent Stream-K lanes: ",
      cudaGetErrorString(result));
  std::vector<uint8_t> host_barriers(barrier_bytes);
  for (at::Tensor const& storage : lane_storage) {
    auto* barrier_ptr = static_cast<uint8_t*>(storage.data_ptr()) +
        barrier_offset;
    result = cudaMemcpy(
        host_barriers.data(), barrier_ptr, barrier_bytes,
        cudaMemcpyDeviceToHost);
    TORCH_CHECK(
        result == cudaSuccess,
        "failed to copy persistent Stream-K barriers: ",
        cudaGetErrorString(result));
    if (!std::all_of(
            host_barriers.begin(), host_barriers.end(),
            [](uint8_t value) { return value == 0; })) {
      return false;
    }
  }
  return true;
}

bool set_workspace_collection_cuda(at::Tensor const& anchor, bool enabled) {
  check_workspace_anchor(anchor);
  bool const previous = collect_workspace_layouts;
  collect_workspace_layouts = enabled;
  return previous;
}

}  // namespace

TORCH_LIBRARY(mxfp6, m) {
  m.def("gemm(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "int m, int n, int k, float alpha=1.0, "
        "ScalarType? out_dtype=None) -> Tensor");
  m.def("gemm_w6a8(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "int m, int n, int k, float alpha=1.0, "
        "ScalarType? out_dtype=None) -> Tensor");
  m.def("gemm_w6a8_config(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "int m, int n, int k, float alpha, int config_id, int swizzle, "
        "int raster_order, ScalarType? out_dtype=None) -> Tensor");
  m.def("gemm_from_float(Tensor input, Tensor b, Tensor sfb, "
        "int n, float alpha=1.0, ScalarType? out_dtype=None) -> Tensor");
  m.def("gemm_from_float_config(Tensor input, Tensor b, Tensor sfb, "
        "int n, float alpha, int config_id, int swizzle, "
        "int raster_order, ScalarType? out_dtype=None) -> Tensor");
  m.def("set_w6a8_config(Tensor anchor, int m, int n, int k, "
        "int config_id, int swizzle, int raster_order, "
        "ScalarType? out_dtype=None) -> bool");
  m.def("w6a8_config_abi(Tensor anchor) -> str");
  m.def("begin_workspace_planning(Tensor anchor) -> ()");
  m.def("finalize_workspace_planning(Tensor anchor) -> Dict(str, int)");
  m.def("workspace_stats(Tensor anchor) -> Dict(str, int)");
  m.def("workspace_barriers_zero(Tensor anchor) -> bool");
  m.def("_set_workspace_collection(Tensor anchor, bool enabled) -> bool");
  m.def("pack_fp6(Tensor codes) -> Tensor");
  m.def("unpack_fp6(Tensor packed, int rows, int k) -> Tensor");
  m.def("expand_fp6_to_fp8(Tensor packed, int rows, int k) -> Tensor");
  m.def("quantize_mxfp8(Tensor input) -> (Tensor values, Tensor scales)");
  m.def("quantize_mxfp6(Tensor input) -> (Tensor values, Tensor scales)");
  m.def("pack_scales(Tensor logical, int rows, int k) -> Tensor");
  m.def("unpack_scales(Tensor packed, int rows, int k) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxfp6, CUDA, m) {
  m.impl("gemm", &gemm_cuda);
  m.impl("gemm_w6a8", &gemm_w6a8_cuda);
  m.impl("gemm_w6a8_config", &gemm_w6a8_config_cuda);
  m.impl("gemm_from_float", &gemm_from_float_cuda);
  m.impl("gemm_from_float_config", &gemm_from_float_config_cuda);
  m.impl("set_w6a8_config", &set_w6a8_config_cuda);
  m.impl("w6a8_config_abi", &w6a8_config_abi_cuda);
  m.impl("begin_workspace_planning", &begin_workspace_planning_cuda);
  m.impl("finalize_workspace_planning", &finalize_workspace_planning_cuda);
  m.impl("workspace_stats", &workspace_stats_cuda);
  m.impl("workspace_barriers_zero", &workspace_barriers_zero_cuda);
  m.impl("_set_workspace_collection", &set_workspace_collection_cuda);
  m.impl("pack_fp6", &mxfp6_gemm::torch_ext::pack_fp6_cuda);
  m.impl("unpack_fp6", &mxfp6_gemm::torch_ext::unpack_fp6_cuda);
  m.impl("expand_fp6_to_fp8",
         &mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda);
  m.impl("quantize_mxfp8", &mxfp6_gemm::torch_ext::quantize_mxfp8_cuda);
  m.impl("quantize_mxfp6", &mxfp6_gemm::torch_ext::quantize_mxfp6_cuda);
  m.impl("pack_scales", &mxfp6_gemm::torch_ext::pack_scales_cuda);
  m.impl("unpack_scales", &mxfp6_gemm::torch_ext::unpack_scales_cuda);
}
