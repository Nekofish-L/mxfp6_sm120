#pragma once
// Shared Stream-K arena implementation. Backend tags keep MXFP6 and MXFP8
// ownership independent while using identical planning and stream-lane rules.
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace mxfp_common {
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

struct WorkspaceSelection {
  at::Tensor owner;
  void* pointer = nullptr;
  bool persistent = false;
};

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

template<class BackendTag>
class WorkspacePool {
  std::mutex workspace_arenas_mutex;
  std::unordered_map<int, WorkspaceArena> workspace_arenas;
 public:
  static bool& collection_enabled() {
    static thread_local bool enabled = true;
    return enabled;
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
        if (collection_enabled()) {
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
    bool const previous = collection_enabled();
    collection_enabled() = enabled;
    return previous;
  }
};
} // namespace mxfp_common
