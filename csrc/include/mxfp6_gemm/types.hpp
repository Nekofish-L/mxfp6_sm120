#pragma once
#include <cstdint>

namespace mxfp6_dispatch {
enum class RasterOrder { Heuristic, AlongM, AlongN };
struct Config { int64_t config_id, swizzle, raster_order; };

inline int64_t round_up(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

inline bool is_target_nk(int64_t n, int64_t k) {
  return (n == 8192 && k == 5120) ||
      (n == 5120 && k == 3072) ||
      (n == 7168 && k == 5120) ||
      (n == 17408 && k == 5120) ||
      (n == 5120 && k == 8704);
}

} // namespace mxfp6_dispatch
