#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
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
#include <torch/custom_class.h>
#include "mxfp6_gemm/prepared_call.hpp"

#include "mxfp6_gemm/w6a8_registry.hpp"
#include "mxfp6_gemm/dispatch.hpp"
#include "mxfp6_gemm/packing.hpp"
#include "mxfp6_gemm/quantization.hpp"
#include "mxfp6_gemm/runtime.hpp"


namespace mxfp6_native {

bool stream_k_enabled() {
  static bool const enabled = [] {
    char const* raw = std::getenv("MXFP6_STREAM_K");
    if (raw == nullptr) {
      return true;
    }
    std::string value(raw);
    auto const not_space = [](unsigned char character) {
      return !std::isspace(character);
    };
    value.erase(
        value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(
        std::find_if(value.rbegin(), value.rend(), not_space).base(),
        value.end());
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char character) {
          return static_cast<char>(std::tolower(character));
        });
    return value != "0" && value != "false";
  }();
  return enabled;
}

using mxfp6_dispatch::RasterOrder;

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

using W6A8LaunchConfig = mxfp6_dispatch::Config;

std::shared_mutex w6a8_overrides_mutex;
std::unordered_map<W6A8ShapeKey, W6A8LaunchConfig, W6A8ShapeKeyHash>
    w6a8_overrides;

mxfp_common::WorkspacePool<MXFP6WorkspaceTag> workspace_pool;

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

using mxfp6_dispatch::round_up;
using mxfp6_dispatch::is_target_nk;

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

thread_local at::ScalarType current_output_dtype = at::kHalf;

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
  TORCH_CHECK(config_id < mxfp6_registry::candidate_count,
              "config_id must be in [0, 29], or negative to erase; got ",
              config_id);
  check_swizzle(swizzle);
  parse_raster_order(raster_order);
  w6a8_overrides[key] = {config_id, swizzle, raster_order};
  return true;
}

std::string w6a8_config_abi_cuda(at::Tensor const& anchor) {
  TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
  return mxfp6_registry::candidate_abi;
}

W6A8LaunchConfig select_w6a8_config(int device_index, int64_t m, int64_t n,
                                   int64_t k, at::ScalarType dtype) {
  W6A8LaunchConfig config{};
  if (!find_w6a8_override(device_index, m, n, k, dtype, config))
    config = mxfp6_dispatch::select_builtin(m, n, k,
        at::cuda::getDeviceProperties(device_index)->multiProcessorCount,
        dtype == at::kHalf, stream_k_enabled());
  return mxfp6_dispatch::normalize(config, stream_k_enabled());
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
  auto config = select_w6a8_config(device_index, m, n, k, current_output_dtype);
  return launch_w6a8_config(a, b, sfa, sfb, m, n, k, alpha, device_index,
                           sm_count, config.config_id, config.swizzle,
                           config.raster_order, use_pdl);
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

  return launch_w6a6_policy(a, b, sfa, sfb, m, n, k, alpha,
                            device_index, properties.multiProcessorCount);
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
                               at::ScalarType output_dtype,
                               bool use_pdl = false) {
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
        use_pdl);
  }
  return launch_w6a8_policy(
      a, b, sfa, sfb, m, n, k, alpha, device_index,
      properties.multiProcessorCount, use_pdl);
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

// Prequantized producer entry; identical dispatch/workspace and PDL policy
// to gemm_from_float. The preceding producer must signal grid readiness or
// finish normally before the consumer's dependency wait can complete.
at::Tensor gemm_w6a8_pdl_cuda(at::Tensor const& a,
                             at::Tensor const& b,
                             at::Tensor const& sfa,
                             at::Tensor const& sfb,
                             int64_t m, int64_t n, int64_t k,
                             double alpha,
                             std::optional<at::ScalarType> output_dtype) {
  return gemm_w6a8_cuda_impl(a, b, sfa, sfb, m, n, k, alpha, -1, 1, 0,
                            resolve_output_dtype(output_dtype), true);
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
                                     at::ScalarType output_dtype,
                                     bool silu_and_mul = false) {
  OutputDtypeGuard output_guard(output_dtype);
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must have shape [M,K]");
  int64_t const m = input.size(0);
  int64_t const k = input.size(1) / (silu_and_mul ? 2 : 1);
  TORCH_CHECK(!silu_and_mul || input.size(1) % 256 == 0,
              "SwiGLU input must have 2K columns with K divisible by 128");
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

  auto quantized = silu_and_mul
      ? mxfp6_gemm::torch_ext::silu_and_mul_mxfp8_cuda(input)
      : mxfp6_gemm::torch_ext::quantize_mxfp8_cuda(input);
  if (config_id >= 0) {
    return launch_w6a8_config(
        std::get<0>(quantized), b, std::get<1>(quantized), sfb,
        m, n, k, alpha, device_index, properties.multiProcessorCount,
        config_id, swizzle, raster_order, true);
  }
  return launch_w6a8_policy(
      std::get<0>(quantized), b, std::get<1>(quantized), sfb,
      m, n, k, alpha, device_index, properties.multiProcessorCount, true);
}

at::Tensor gemm_from_swiglu_cuda(at::Tensor const& input,
                                at::Tensor const& b,
                                at::Tensor const& sfb,
                                int64_t n,
                                double alpha,
                                std::optional<at::ScalarType> output_dtype) {
  return gemm_from_float_cuda_impl(
      input, b, sfb, n, alpha, -1, 1, 0, resolve_output_dtype(output_dtype), true);
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


void begin_workspace_planning_cuda(at::Tensor const& anchor) {
  workspace_pool.begin_workspace_planning_cuda(anchor);
}
c10::Dict<std::string, int64_t> finalize_workspace_planning_cuda(at::Tensor const& anchor) {
  return workspace_pool.finalize_workspace_planning_cuda(anchor);
}
c10::Dict<std::string, int64_t> workspace_stats_cuda(at::Tensor const& anchor) {
  return workspace_pool.workspace_stats_cuda(anchor);
}
bool workspace_barriers_zero_cuda(at::Tensor const& anchor) {
  return workspace_pool.workspace_barriers_zero_cuda(anchor);
}
bool set_workspace_collection_cuda(at::Tensor const& anchor, bool enabled) {
  return workspace_pool.set_workspace_collection_cuda(anchor, enabled);
}

thread_local PreparedLaunch* preparing = nullptr;

PreparedW6A8::PreparedW6A8(at::Tensor a, at::Tensor b, at::Tensor sfa, at::Tensor sfb,
              int64_t m, int64_t n, int64_t k, double alpha,
              int64_t config_id, int64_t swizzle, int64_t raster_order,
              at::ScalarType dtype, bool pdl) : pdl_(pdl) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  c10::cuda::CUDAGuard guard(a.device());
  auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
              "prepare_w6a8 must run before CUDA Graph capture");
  config_ = config_id >= 0 ? W6A8LaunchConfig{config_id, swizzle, raster_order}
      : select_w6a8_config(a.get_device(), m, n, k, dtype);
  config_ = mxfp6_dispatch::normalize(config_, stream_k_enabled());
  struct PreparationGuard {
    PreparedLaunch* previous;
    explicit PreparationGuard(PreparedLaunch* current) : previous(preparing) { preparing = current; }
    ~PreparationGuard() { preparing = previous; }
  } preparation_guard(&launch_);
  auto out = gemm_w6a8_cuda_impl(a, b, sfa, sfb, m, n, k, alpha,
      config_.config_id, config_.swizzle, config_.raster_order, dtype, pdl);
  TORCH_CHECK(static_cast<bool>(launch_.run), "missing prepared kernel");
  tensors_ = {a, b, sfa, sfb, out};
  for (auto const& t : tensors_)
    snapshots_.push_back({t.data_ptr(), t.sizes().vec(), t.strides().vec(),
                          t.scalar_type(), t.device()});
  // Publish initialized workspace to later streams without a device sync.
  // Preparation is cold; one stream sync avoids an event in every hot call.
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}
at::Tensor PreparedW6A8::run() {
  c10::cuda::CUDAGuard guard(snapshots_[0].device);
  for (size_t i = 0; i < tensors_.size(); ++i) {
    auto const& t = tensors_[i]; auto const& s = snapshots_[i];
    TORCH_CHECK(t.device() == s.device && t.scalar_type() == s.dtype &&
                t.data_ptr() == s.pointer && t.sizes() == at::IntArrayRef(s.sizes) &&
                t.strides() == at::IntArrayRef(s.strides),
                "Prepared W6A8 tensor storage, shape, strides and dtype must remain unchanged");
  }
  auto stream = c10::cuda::getCurrentCUDAStream(snapshots_[0].device.index());
  launch_.used = true;
  launch_.reset_workspace(stream);
  launch_.run(stream, pdl_);
  for (auto const& t : tensors_)
    c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), stream);
  if (launch_.workspace.defined())
    c10::cuda::CUDACachingAllocator::recordStream(launch_.workspace.storage().data_ptr(), stream);
  return tensors_.back();
}

c10::intrusive_ptr<PreparedW6A8> prepare_w6a8_cuda(
    at::Tensor a, at::Tensor b, at::Tensor sfa, at::Tensor sfb,
    int64_t m, int64_t n, int64_t k, double alpha, int64_t config_id,
    int64_t swizzle, int64_t raster_order, std::optional<at::ScalarType> dtype, bool pdl) {
  return c10::make_intrusive<PreparedW6A8>(a, b, sfa, sfb, m, n, k, alpha,
      config_id, swizzle, raster_order, resolve_output_dtype(dtype), pdl);
}

}  // namespace mxfp6_native

using namespace mxfp6_native;

TORCH_LIBRARY(mxfp6, m) {
  m.class_<PreparedW6A8>("PreparedW6A8")
      .def("run", &PreparedW6A8::run)
      .def("config", &PreparedW6A8::config);
  m.def("prepare_w6a8(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "int m, int n, int k, float alpha, int config_id, int swizzle, "
        "int raster_order, ScalarType? out_dtype, bool pdl) "
        "-> __torch__.torch.classes.mxfp6.PreparedW6A8");
  m.def("gemm(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "int m, int n, int k, float alpha=1.0, "
        "ScalarType? out_dtype=None) -> Tensor");
  m.def("gemm_w6a8(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "int m, int n, int k, float alpha=1.0, "
        "ScalarType? out_dtype=None) -> Tensor");
  m.def("gemm_w6a8_config(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "int m, int n, int k, float alpha, int config_id, int swizzle, "
        "int raster_order, ScalarType? out_dtype=None) -> Tensor");
  m.def("gemm_w6a8_pdl(Tensor a, Tensor b, Tensor sfa, Tensor sfb, "
        "int m, int n, int k, float alpha=1.0, ScalarType? out_dtype=None) -> Tensor");
  m.def("gemm_from_swiglu(Tensor input, Tensor b, Tensor sfb, "
        "int n, float alpha=1.0, ScalarType? out_dtype=None) -> Tensor");
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
  m.def("pad_fp6(Tensor packed) -> Tensor");
  m.def("expand_fp6_to_fp8(Tensor packed, int rows, int k) -> Tensor");
  m.def("quantize_mxfp8(Tensor input) -> (Tensor values, Tensor scales)");
  m.def("quantize_mxfp8_dual(Tensor input) -> "
        "(Tensor values, Tensor logical_scales, Tensor packed_scales)");
  m.def("quantize_mxfp8_dual_out("
        "Tensor(a!) output, Tensor(b!) logical_scales, "
        "Tensor(c!) packed_scales, Tensor input) -> ()");
  m.def("quantize_mxfp6(Tensor input) -> (Tensor values, Tensor scales)");
  m.def("pack_scales(Tensor logical, int rows, int k) -> Tensor");
  m.def("unpack_scales(Tensor packed, int rows, int k) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxfp6, CUDA, m) {
  m.impl("prepare_w6a8", &prepare_w6a8_cuda);
  m.impl("gemm", &gemm_cuda);
  m.impl("gemm_w6a8", &gemm_w6a8_cuda);
  m.impl("gemm_w6a8_config", &gemm_w6a8_config_cuda);
  m.impl("gemm_from_float", &gemm_from_float_cuda);
  m.impl("gemm_w6a8_pdl", &gemm_w6a8_pdl_cuda);
  m.impl("gemm_from_swiglu", &gemm_from_swiglu_cuda);
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
  m.impl("pad_fp6", &mxfp6_gemm::torch_ext::pad_fp6_cuda);
  m.impl("expand_fp6_to_fp8",
         &mxfp6_gemm::torch_ext::expand_fp6_to_fp8_cuda);
  m.impl("quantize_mxfp8", &mxfp6_gemm::torch_ext::quantize_mxfp8_cuda);
  m.impl("quantize_mxfp8_dual",
         &mxfp6_gemm::torch_ext::quantize_mxfp8_dual_cuda);
  m.impl("quantize_mxfp8_dual_out",
         &mxfp6_gemm::torch_ext::quantize_mxfp8_dual_out_cuda);
  m.impl("quantize_mxfp6", &mxfp6_gemm::torch_ext::quantize_mxfp6_cuda);
  m.impl("pack_scales", &mxfp6_gemm::torch_ext::pack_scales_cuda);
  m.impl("unpack_scales", &mxfp6_gemm::torch_ext::unpack_scales_cuda);
}
